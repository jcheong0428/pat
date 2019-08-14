from __future__ import division

'''
PAT Tools
===========================
Tools to help analysis of OpenPose data.
'''

__all__ = ['load_keypoints']

import json, os, glob
import pandas as pd, numpy as np

def load_keypoints(fname, frame_no):
    '''Load OpenPose keypoints output
    Args:
        fname: filename of OpenPose output keypoints
        frame_no: frame number to be appended to file

    Returns:
        Pandas dataframe that includes filename, inferred frameno, value, key, keyID.

    Example Use:
        import glob, os
        import pandas as pd
        from tqdm import tqdm

        fnames = np.sort(glob.glob('output/json/*_keypoints.json'))
        new_df_fname = 'output/Sherlock.csv'
        if not os.path.exists(new_df_fname):
            for fname in tqdm(fnames[:5000]):
                frame_no = os.path.split(fname)[1].split('_')[1]
                load_keypoints(fname, frame_no = frame_no).to_csv(new_df_fname, index=False, header=False, mode='a')
        else:
            col_names = ['fname', 'frame', 'key','keyID', 'personID','value']
            df = pd.read_csv(new_df_fname, header=None, names=col_names)
    '''
    with open(fname) as json_file:
        data = json.load(json_file)
    # check if no_people different from number of unique people ids
    no_people = len(data['people'])
    people_ids = [data['people'][i_people]['person_id'][0] for i_people in range(no_people)]
    if no_people != len(np.unique(people_ids)):
        people_ids = list(range(no_people))
    df = pd.DataFrame()
    for i_people in range(no_people):
        for key in data['people'][i_people].keys():
            value = data['people'][i_people][key]
            df = pd.concat([df, pd.DataFrame({'fname': fname, 'frame': frame_no,
                                            'key': key, 'keyID': [f"{key}_{str(i).zfill(3)}" for i in range(len(value))],
                                            'personID': people_ids[i_people],
                                            'value': value
                                            })])
    return df.reset_index(drop=True)

def calc_time(frame, fps):
    """Calculates minutes:seconds (mm:ss) from frame number given frames per second
    Args:
        frame: float
            frame number
        fps: float
            frames per second
    Return:
        time: string
            movie time in mm:ss format
    """
    fps = 25.
    time = int(frame)/fps/60.
    time_min = int(np.floor(time))
    time_sec = int(np.round(60*float(str(time-int(time))[1:])))
    return f"{time_min}:{time_sec}"

def load_keypoints_2d(fname, frame_no, out_fname):
    '''Load OpenPose pose 2d keypoints output
    Args:
      fname: filename of OpenPose output keypoints
      frame_no: frame number to be appended to file
      out_fname: filename for output

    Returns:
      Pandas dataframe that includes filename, inferred frameno, value, key, keyID.
    '''
    with open(fname) as json_file:
        data = json.load(json_file)
    # check if no_people different from number of unique people ids  
    no_people = len(data['people'])
    if no_people > 0:
        people_ids = [data['people'][i_people]['person_id'][0] for i_people in range(no_people)]
        if no_people != len(np.unique(people_ids)):
            people_ids = list(range(no_people))
        df = pd.DataFrame()
        for i_people in range(no_people):
            for key in data['people'][i_people].keys():
                value = data['people'][i_people][key]
                df = pd.concat([df, pd.DataFrame({'fname': fname, 'frame': frame_no, 'value': value, 'personID': people_ids[i_people] ,
                                                'key': key, 'keyID': [f"{key}_{str(i).zfill(3)}" for i in range(len(value))]
                                                })])
        df = df.reset_index(drop=True).pat.grab_pose().pat.grab_person_pose()
        df.to_csv(out_fname, index=True, header=False, mode='a')
