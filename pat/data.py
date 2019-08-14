from __future__ import division

'''
Pose Analysis Toolbox PAT Data Class
===========================
Class extending pandas to represent and manipulate OpenPose data.
'''
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist, squareform
import matplotlib.pyplot as plt
from pat.utils import calc_time

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
        self._validate_keypoints(pandas_obj)
        self._obj = pandas_obj
        self._colnames = ['fname', 'frame', 'key','keyID', 'personID','value']
        self._type = 'Keypoints'
        self.fps = None

    @staticmethod
    def _validate_keypoints(obj):
        '''Verify that object is a keypoints dataframe
        '''
        for col in obj.columns:
            if col not in obj.columns:
                raise AttributeError(f"Must have {col} in columns.")
        pass

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
        return self._obj.groupby('frame')['personID'].nunique()

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
            return_df.pat._type = 'Pose2D'
            '''
            TODO: change column names to something meaningful?
            TODO: set attribute that indicates it's pose data?
            '''
            return return_df
        else:
            '''
            TODO: change column names to something meaningful?
            TODO: set attribute that indicates it's pose data?
            '''
            self._type = 'Pose2D'
            return self._obj.query('personID==@personID').pivot(index='frame',columns='keyID',values='value')

    def extract_distance(self, metric='euclidean'):
        '''Extracts distance matrix (DM) for each keypoint at each index.
        It does not calculate a DM for background, so returns a 276 columns back (24*23/2=(n*(n-1)/2))

        Args:
            self: pose2D, dataframe with 75 column keypoint (includes background).
            metric: distance metric used in scipy.spatial.distance.cdist (default: euclidean)
        Returns:
            Dataframe with frame x keypoint distance matrix ()
        '''
        assert(self._type=='Pose2D'), "Make sure dataframe is a Pose2D dataframe with 75 columns."
        pose_dms = []
        for frame_ix in range(len(self._obj)):
            p_x = self._obj.iloc[frame_ix,:72:3]
            p_y = self._obj.iloc[frame_ix,1:72:3]
            pose_dms.append(squareform(cdist(np.array([p_x, p_y]).T,np.array([p_x, p_y]).T, metric=metric)))
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
        assert(self._type=='Pose2D'), "Make sure dataframe is a Pose2D dataframe with 75 columns."
        '''
        TODO: Extend to filtering per value.
        '''
        filter_bool = []
        for frame_ix in range(len(self._obj)):
            p_c = self._obj.iloc[frame_ix,2:72:3]
            filter_bool.append(filter_func(p_c) > min_conf)
        return self._obj.loc[filter_bool]

    def plot(self, frame_no = None, ax = None, xlim = [0,640], ylim = [0,480], **kwargs):
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
        assert(self._type=='Pose2D'), "Make sure dataframe is a Pose2D dataframe with 75 columns."
        assert(isinstance(frame_no, (int, float))), "Make sure your frame_no is a number"
        frame_df = self._obj.query("frame==@frame_no")
        xs = frame_df.iloc[:,:72:3]
        ys = frame_df.iloc[:,1:72:3]
        if ax is None:
            f,ax = plt.subplots()
        ax.scatter(xs, ys, **kwargs)
        ax.axes.set_xscale('linear')
        ax.axes.set_yscale('linear')
        ax.set(xlim=xlim, ylim=ylim, xticks=[], xticklabels=[], yticklabels=[])

        if self.fps:
            title = f"Frame: {frame_no} Time: {calc_time(frame_no, fps=self.fps)}"
        else:
            title = f"Frame: {frame_no}"
        ax.set(ylim=ax.get_ylim()[::-1], title=title)
        return ax
