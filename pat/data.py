from __future__ import division

'''
Pose Analysis Toolbox PAT Data Class
===========================
Class extending pandas to represent and manipulate OpenPose data.
'''
import pandas as pd
import numpy as np
import os, glob
from scipy.spatial.distance import cdist, squareform
import matplotlib.pyplot as plt
from pat.utils import calc_time, get_resource_path
from scipy.spatial import procrustes
from tqdm import tqdm
import h5py

pose_2d_keys =  {0:  "Nose", 1:  "Neck", 2: "RShoulder",
3: "RElbow", 4: "RWrist", 5: "LShoulder", 6: "LElbow",
7: "LWrist", 8: "MidHip", 9: "RHip",
10: "RKnee", 11: "RAnkle", 12: "LHip", 13: "LKnee", 14: "LAnkle", 15: "REye",
16: "LEye", 17: "REar", 18: "LEar", 19: "LBigToe", 20: "LSmallToe",
21: "LHeel", 22: "RBigToe", 23: "RSmallToe", 24: "RHeel", 25: "Background"}
pose2d_cols = np.ravel([[f'x_{pose_2d_keys[i]}',
                         f'y_{pose_2d_keys[i]}',
                         f'c_{pose_2d_keys[i]}'] for i in range(25)])
standardfigure = pd.read_csv(os.path.join(get_resource_path(),'standardfig.csv'),index_col=['frame'])

@pd.api.extensions.register_dataframe_accessor("pat")
class PoseAnalysisToolbox:
    '''
    PoseAnalysisToolbox "pat" is an extension of pandas dataframe to represent and manipulate OpenPose data.

    Args:
        pandas_obj: pandas dataframe
            dataframe instance loaded with utils.load_keypoints, or one that includes columns [fname, frame, key, keyID, personID, value]
        fps: float, default none
            frames per second of original data.

    Attributes:
        _obj: pandas dataframe object
        _colnames: column names .. prob not needed
        _type: data structure type
            Keypoints: Long format dataframe that reads in ALL Keypoints information loaded from OpenPose
            Pose2D: Time x 75 matrix in which columns refer to
                https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md
        fps: frames per second of the video from which the pose data were extracted.

    '''
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        self._type = self._validate_type(pandas_obj)
        self.fps = None

    @staticmethod
    def _validate_type(obj):
        '''Verify that object is a keypoints dataframe
        '''
        keypoint_cols = ['fname', 'frame', 'key', 'keyID', 'personID', 'value']
        type = ''
        if np.array_equal(obj.columns, keypoint_cols):
            type = 'Keypoints'
        elif np.array_equal(obj.columns, pose2d_cols):
            type = 'Pose2D'
        return type

    def grab_pose(self):
        '''Grabs the pose_keypoints_2d
        Args:
            df: Keypoints dataframe
        Returns:
            pose_keypoints_2d: pandas dataframe
        '''
        assert(self._type=='Keypoints'), "Make sure dataframe is in Keypoints format"
        return self._obj.query(f"key=='pose_keypoints_2d'")

    def grab_people(self):
        '''Grabs the number of people in each frame
        Args:
            df: Keypoints dataframe
        Returns:
            person_counts: pandas series
        '''
        if self._type=='Keypoints':
            return self._obj.groupby('frame')['personID'].nunique()
        else:
            return self._obj.groupby('frame').count().mean(axis=1)

    def grab_person_pose(self, personID=None):
        '''Grabs the pose keypoint data for specified personid.
        Args:
            self: Long format keypoints dataframe
            personID: ID of person whose pose will be extracted. If None, extract for everyone (default, None)
        Returns:
            Dataframe for specified personID
        '''
        if (len(self._obj.key.unique()) != 1) or (self._obj.key.unique()[0] != 'pose_keypoints_2d'):
            self = self.grab_pose()

        if personID==None:
            return_df = pd.DataFrame()
            for personID in self._obj.personID.unique():
                person_pose_df = self._obj.query('personID==@personID').pivot(index='frame',columns='keyID',values='value')
                person_pose_df['personID'] = personID
                return_df = pd.concat([return_df,
                                       person_pose_df
                                      ],axis=0)
            return_df = return_df.set_index(keys = ['personID'], drop = True, append = True)
            del return_df.columns.name
        else:
            return_df = self._obj.query('personID==@personID').pivot(index='frame',columns='keyID',values='value')
        self._type = 'Pose2D'
        return_df.columns = pose2d_cols
        return return_df

    def extract_distance(self, metric='euclidean', override=False):
        '''Extracts distance matrix (DM) for each keypoint at each index.
        It does not calculate a DM for background, so returns a 276 columns back (24*23/2=(n*(n-1)/2))

        Args:
            self: pose2D, dataframe with 75 column keypoint (includes background).
            metric: distance metric used in scipy.spatial.distance.cdist (default: euclidean)
            override: override data form check.
        Returns:
            Dataframe with frame x keypoint distance matrix ()
        '''
        xcols = [np.where(col==self._obj.columns)[0][0] for col in self._obj.columns if 'x_' in col]
        ycols = [np.where(col==self._obj.columns)[0][0] for col in self._obj.columns if 'y_' in col]
        if len(xcols)==0:
            raise KeyError("No confidence columns beginning with x_")
        if len(ycols)==0:
            raise KeyError("No confidence columns beginning with c_")

        pose_dms = []
        for frame_ix in range(len(self._obj)):
            p_x = self._obj.iloc[frame_ix,xcols]
            p_y = self._obj.iloc[frame_ix,ycols]
            coords = np.array([p_x, p_y]).T
            pose_dms.append(squareform(cdist(coords, coords, metric=metric)))
        return pd.DataFrame(pose_dms, index=self._obj.index)

    def filter_pose_confidence(self, min_conf=.2, filter_func = np.mean):
        '''Filter pose_2d keypoints by confidence.
        Args:
            self: pose2D, dataframe with 75 column keypoint (includes background).
            min_conf: minimum confidence per frame to remove
            filter_func: function to filter confidence per frame (default: np.mean)
        Return:
            filtered Dataframe
        '''
        # assert(self._type=='Pose2D'), "Make sure dataframe is a Pose2D dataframe with 75 columns."
        ccols = [np.where(col==self._obj.columns)[0][0] for col in self._obj.columns if 'c_' in col]
        if len(ccols)==0:
            raise KeyError("No confidence columns beginning with c_")
        '''
        TODO: Extend to filtering per value.
        '''
        filter_bool = []
        for frame_ix in range(len(self._obj)):
            p_c = self._obj.iloc[frame_ix,ccols]
            filter_bool.append(filter_func(p_c) > min_conf)
        return self._obj.loc[filter_bool]

    def plot(self, frame_no = None, ax = None, xlim = None, ylim = None, title=None,**kwargs):
        """Plots the Pose2D data
        Args:
            frame_no: int, float
                frame number that corresponds to the pose2D dataframe index
            xlim: minimum to maximum x coordinates of screensize
            ylim: minimum to maximum y coordinates of screensize
            **kwargs: arguments to pass to scatter.
        Return:
            ax: matplotlib ax handle
        """
        # assert(self._type=='Pose2D'), "Make sure dataframe is a Pose2D dataframe with 75 columns."
        # assert(isinstance(frame_no, (int, float))), "Make sure your frame_no is a number"
        if len(self._obj)!=1:
            try:
                frame_df = self._obj.query("frame==@frame_no")
            except:
                raise("Please specifify which frame to plot")
        else:
            frame_df = self._obj

        xcols = [np.where(col==self._obj.columns)[0][0] for col in self._obj.columns if 'x_' in col]
        ycols = [np.where(col==self._obj.columns)[0][0] for col in self._obj.columns if 'y_' in col]
        if len(xcols)==0:
            raise KeyError("No confidence columns beginning with x_")
        if len(ycols)==0:
            raise KeyError("No confidence columns beginning with c_")

        xs = frame_df.iloc[:,xcols]
        ys = frame_df.iloc[:,ycols]
        if ax is None:
            f,ax = plt.subplots()
        ax.scatter(xs, ys, **kwargs)
        ax.axes.set_xscale('linear')
        ax.axes.set_yscale('linear')
        if xlim is not None:
            ax.set(xlim=xlim)
        if ylim is not None:
            ax.set(ylim=ylim)
        ax.set(xticks=[], xticklabels=[], yticklabels=[])

        if title is None:
            if self.fps:
                title = f"Frame: {frame_no} Time: {calc_time(frame_no, fps=self.fps)}"
            else:
                title = f"Frame: {frame_no}"
        ax.set(ylim=ax.get_ylim()[::-1], title=title)
        return ax

    def align(self, standardfigure = None):
        '''Uses Procrustes transformation to align to standard figure
        If you have nans in your data, then it will try to use subset of keypoints that are not nans.

        Args:
            standardfigure: dataframe in Pose2D format to act as standard. default, standardfig.csv
        Returns:
            df: Pose2D dataframe with aligned coordinates.
        '''
        def _grab_coordinates(df):
            '''Grabs the x, y coordinates and spits out n x (x,y) matrix

            Args:
                df: Pose2D dataframe
            Returns:
                df: Pose2D dataframe with x, y coordinates.
            '''
            newdf = df[[col for col in df.columns if 'x_' in col or 'y_' in col]]
            return newdf.values.reshape(int(newdf.shape[1]/2),2)

        aligned_df = self._obj.copy()
        if standardfigure is None:
            standardfigure = pd.read_csv(os.path.join(get_resource_path(),'standardfig.csv'),index_col=['frame'])
            xy_standard = _grab_coordinates(standardfigure)
        for rowix, row in self._obj.iterrows():
            # check if nans exist
            coords = _grab_coordinates(row.to_frame().T)
            coordbool = np.any(~np.isnan(coords),axis=1)
            if np.any(~coordbool):
                mtx1, mtx2, _ = procrustes(xy_standard[coordbool], coords[coordbool])
                coords[coordbool] = mtx2
                aligned_df.loc[rowix, [col for col in self._obj.columns if 'x_' in col or 'y_' in col]] = coords.flatten()
            else:
                mtx1, mtx2, _ = procrustes(xy_standard, coords)
                aligned_df.loc[rowix, [col for col in self._obj.columns if 'x_' in col or 'y_' in col]] = mtx2.flatten()
        return aligned_df

    def impute(self):
        """Impute missing data.
        You should specify what data are missing. Preferabley by setting 0 values to np.nans.

        Args:
            nothing
        Returns:
            imputed dataframe
        """
        with h5py.File(os.path.join(get_resource_path(),"impute_weights.hdf5"), "r") as f:
            sq_weight_x = np.array(f['sq_weight_x'])
            sq_weight_y = np.array(f['sq_weight_y'])

        xcols = [col for col in self._obj.columns if 'x_' in col]
        ycols = [col for col in self._obj.columns if 'y_' in col]

        imputed_df = self._obj.copy()
        for rowix, row in tqdm(self._obj.iterrows(), total=self._obj.shape[0]):
            xs = row[xcols].values
            ys = row[ycols].values
            for ix, x in enumerate(xs):
                if np.isnan(x):
                    notnull = ~np.isnan(xs)
                    norm_wm = 1 - sq_weight_x[ix][notnull]/sq_weight_x[ix][notnull].sum()
                    imputed_value = np.mean(xs[notnull]*norm_wm)
                    imputed_df.loc[rowix, xcols[ix]] = imputed_value
            for iy, y in enumerate(ys):
                if np.isnan(y):
                    notnull = ~np.isnan(ys)
                    norm_wm = 1 - sq_weight_y[iy][notnull]/sq_weight_x[iy][notnull].sum()
                    imputed_value = np.mean(ys[notnull]*norm_wm)
                    imputed_df.loc[rowix, ycols[iy]] = imputed_value
        return imputed_df
