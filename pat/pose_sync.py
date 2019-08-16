import os 
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import csv
import random
import logging
from tqdm import tqdm

# from data import pose2d_cols
pose_2d_keys =  {0:  "Nose", 1:  "Neck", 2: "RShoulder",
3: "RElbow", 4: "RWrist", 5: "LShoulder", 6: "LElbow",
7: "LWrist", 8: "MidHip", 9: "RHip",
10: "RKnee", 11: "RAnkle", 12: "LHip", 13: "LKnee", 14: "LAnkle", 15: "REye",
16: "LEye", 17: "REar", 18: "LEar", 19: "LBigToe", 20: "LSmallToe",
21: "LHeel", 22: "RBigToe", 23: "RSmallToe", 24: "RHeel", 25: "Background"}
pose2d_cols = np.ravel([[f'x_{pose_2d_keys[i]}',
                         f'y_{pose_2d_keys[i]}',
                         f'c_{pose_2d_keys[i]}'] for i in range(25)])


PERSON_ID_START = 1000

def min_dist_between_people(person_pose, people_poses):
    """
    Minumum euclidean distance between one person and a bunch of people
    Args:
        person_pose: (pandas DataFrame) df containing one set of pose keypoints
        people_poses: (pandas DataFrame) df containing sets of pose keypoints to
                      be compared with
    """
    # TODO forgive/ignore it if one point disappear
    all_dists = [np.mean(cdist(list(person_pose.to_numpy()), [list(other_pose)]))
                 for _, other_pose in people_poses.iterrows()]
    if 'logger' in globals():
        for dist in all_dists:
            logger.info(dist)
    index = np.argmin(all_dists)
    min_dist = all_dists[index]
    min_dist_personID = people_poses.index.get_level_values(1)[index]
    return min_dist, min_dist_personID


def identity_sync(df, start_frame=0, end_frame=None, threshold=350):
    '''
    Identify the same person across frames, and change the personID accordingly.
    Successfully identified people will have integers starting from 1000 as their
    new IDs.
    Args:
        df: (pandas DataFrame) keypoints dataframe (frame x keypoints)
        start_frame: (integer) start frame number (included)
        end_frame: (integer) end frame number (not included)
        threshold: (float) the maximum distance between two poses that are still
                   considered belonging to the same person
    Returns:
        Dataframe from start_frame to end_frame, with updated perseonID
    '''
    # get (partial) data frame
    person_id_tracker = PERSON_ID_START
    id_df = df[df.index.get_level_values(0) >= start_frame]
    if end_frame:
        id_df = id_df[id_df.index.get_level_values(0) < end_frame]
    id_df = id_df.sort_values(['frame', 'personID'])

    current_people_i = []  # personIDs in current frame
    all_new_indexes = []  # new list of tuples to replace the old MultiIndex

    for frame, frame_df in tqdm(id_df.groupby(level=0)):
        if (frame + 1) not in id_df.index.get_level_values(0):  # next frame doesn't exist
            if len(current_people_i) > 0:
                # use indexes that were updated in previous loop
                all_new_indexes += [(frame, p) for p in current_people_i]
            else:  # just use existing indexes
                all_new_indexes += frame_df.index.tolist()
            current_people_i = []
            continue

        if len(current_people_i) == 0:
            current_people_i = frame_df.index.get_level_values(1)
        next_frame_df = id_df.loc[[frame + 1]]
        next_people_i = list(next_frame_df.index.get_level_values(1))
        for i, (person, keypoints) in enumerate(frame_df.groupby(level=1)):
            min_dist, next_person_i = min_dist_between_people(keypoints, next_frame_df)
            if min_dist < threshold:  # matched a person
                person_id = current_people_i[i]
                if person_id < PERSON_ID_START:  # current person was unknown
                    # replace current personID with new ID
                    person_id = person_id_tracker
                    current_people_i = list(map(lambda x: person_id if x == person else x,
                                                current_people_i))
                    person_id_tracker += 1
                # also update personID in the next frame
                next_people_i = list(map(lambda x: person_id if x == next_person_i else x,
                                         next_people_i))
        all_new_indexes += [(frame, p) for p in current_people_i]
        current_people_i = next_people_i

    all_new_indexes = pd.MultiIndex.from_tuples(all_new_indexes, names=('frame', 'personID'))
    id_df.index = all_new_indexes
    return id_df


def plot_distance_distribution(pose_file, dist_file):
    '''
    TODO make this a jupyter notebook example?
    '''

    df = pd.read_csv(pose_file, header=None, index_col=[0, 1], names=pose2d_cols)
    df.index.names = ['frame','personID']
    df = df.sort_values(['frame','personID'])
    start_frames = random.sample([1000 * i for i in range(0, 75)], 10)
    identity_sync(df, start_frame=13879, end_frame=13962)
    for f in start_frames:
        identity_sync(df, start_frame=f, end_frame=f + 1000)

    import matplotlib.pyplot as plt
    dist = np.loadtxt(dist_file)
    fig, ax = plt.subplots()
    ax.hist(dist, 1000, range=[0, 1000], density=True, color='r')
    plt.show()


def test_identity_sync():
    '''
    TODO put this in tests
    '''
    df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3],
                       [7, 8, 9], [1, 2, 3]])
    df.index = pd.MultiIndex.from_tuples(
        [(123, -1), (124, 0), (124, 1), (126, 0), (126, 1), (126, 2), (127, 0), (127, 1)],
        names=('frame', 'personID'))
    new_df = identity_sync(df, threshold=1)
    np.testing.assert_array_equal(new_df.index.get_level_values(1),
                                  [1000, 0, 1000, 0, 1001, 1002, 1001, 1002])
    print(new_df)

    df = pd.DataFrame([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
    df.index = pd.MultiIndex.from_tuples(
        [(123, -1), (124, -1), (125, -1), (126, -1), (127, -1)],
        names=('frame', 'personID'))
    new_df = identity_sync(df, threshold=1)
    np.testing.assert_array_equal(new_df.index.get_level_values(1),
                                  [1000, 1000, 1000, 1000, 1000])
    print(new_df)

    df = pd.DataFrame([[1, 2, 3], [1, 2, 5], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
    df.index = pd.MultiIndex.from_tuples(
        [(123, -1), (123, 0), (124, -1), (125, -1), (126, -1)],
        names=('frame', 'personID'))
    new_df = identity_sync(df, threshold=1)
    np.testing.assert_array_equal(new_df.index.get_level_values(1),
                                  [1000, 0, 1000, 1000, 1000])
    print(new_df)


if __name__ == '__main__':
    # TODO make this a jupyter notebook example?
    handler = logging.FileHandler('distance_data.csv')
    logger = logging.getLogger('data_log')
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    plot_distance_distribution('../notebooks/output/Sherlock_full_par.csv', 'distance_data.csv')

    # df = pd.read_csv('Sherlock_full_par.csv',
    #                  header=None, index_col=[0, 1], names=pose2d_cols)
    # df.index.names = ['frame','personID']
    # df = df.sort_values(['frame','personID'])
    # identity_sync(df).to_csv('sherlock_pose_sync.csv')
