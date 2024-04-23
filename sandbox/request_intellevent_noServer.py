# Note: tensorflow, numpy and pandas MUST have the described versions
# otherwise the mode cannot be opened and the code won´t work.

from viconnexusapi import ViconNexus
# cd C:\Program Files\Vicon\Nexus2.14\SDK\Win64\Python
#pip install ./viconnexusapi
from sklearn import preprocessing
import numpy as np #conda install numpy==1.18.5
#from ezc3d import c3d #conda install -c conda-forge ezc3d
#from utils import reshape_data, fc_pred, fo_pred
import time
import threading
import pandas as pd #conda install pandas==1.2.0
from pyCGM2.Tools import btkTools

import tensorflow as tf
from scipy.signal import find_peaks


MIN_PEAK_THRESHOLD = 0.2


def reshape_data(traj):
    rs_traj = np.transpose(np.array(traj).reshape(1, traj.shape[0], traj.shape[1]), (0, 2, 1))
    return rs_traj


# marker names which are used for the algorithm
# adapt names to how they are stored in your .c3d file
marker_list = ["LHEE", "LTOE", "LANK", "RHEE", "RTOE", "RANK"]
n_markers = len(marker_list)

if __name__=='__main__':
    # timer = time.time()

    # Get Vicon Nexus specific information from the viconnexus API
    vicon = ViconNexus.ViconNexus()
    path, file_name = vicon.GetTrialName()
    
    
    #  #----ezc3d------------   
    # # opens the current .c3d file which is open in Vicon Nexus
    #  c3d_trial = c3d(path + file_name + '.c3d', extract_forceplat_data=True)
    subject_name = vicon.GetSubjectNames()
    

    # trial_information = c3d_trial['parameters']['TRIAL']
    # start_frame = trial_information['ACTUAL_START_FIELD']['value'][0]

    # # Get all marker label names from the .c3d file
    # labels = c3d_trial['parameters']['POINT']['LABELS']['value']
    # # Get the corresponding index for each marker name in 'marker_list'
    # marker_index = [labels.index(label) for label in marker_list]

    # # Get all trajectories from the .c3d file
    # trajectory_list = c3d_trial['data']['points']

    # # Get the x, y, and z trajectories corresponding to the 'marker_index' list
    # # Note: x-axis = direction of movement, y-axis = mediolateral movement, z-axis = vertical movement
    # x_traj = trajectory_list[0, marker_index, :]
    # y_traj = trajectory_list[1, marker_index, :]
    # z_traj = trajectory_list[2, marker_index, :]

    # # The current best model uses the x and z axis velocity for the IC model
    # # and the x, y, and z axis velocity for the FO model
    # fc_traj = np.concatenate([x_traj, z_traj])
    # fo_traj = np.concatenate([x_traj, y_traj, z_traj])
    # #--------------


    acq =  btkTools.smartReader(path + file_name + '.c3d')


    start_frame = acq.GetFirstFrame()

    array = btkTools.markersToArray(acq,marker_list,gathercolumn=True)
    fc_traj = np.concatenate([array[:,0:n_markers], array[:,2*n_markers:3*n_markers]],axis=1).T
    fo_traj = array.T 


    # x and y-coordinates need to be standardized depending on the starting direction,
    # z coordinates are always the same
    if any(fc_traj[0, 0:10] < 0):
        fc_traj[0:6, :] = (fc_traj[0:6, :] - np.mean(fc_traj[0:6, :], axis=1).reshape(6,1)) * (-1)
        fo_traj[0:12, :] = (fo_traj[0:12, :] - np.mean(fo_traj[0:12, :], axis=1).reshape(12, 1)) * (-1)
    else:
        fc_traj[0:6, :] = fc_traj[0:6, :] - np.mean(fc_traj[0:6, :], axis=1).reshape(6,1)
        fo_traj[0:12, :] = fo_traj[0:12, :] - np.mean(fo_traj[0:12, :], axis=1).reshape(12, 1)

    fc_traj[6:12, :] = fc_traj[6:12, :] - np.mean(fc_traj[6:12, :], axis=1).reshape(6,1)
    fo_traj[12:18, :] = fo_traj[12:18, :] - np.mean(fo_traj[12:18, :], axis=1).reshape(6,1)

    # calculate the first derivative (= velocity) of the trajectories
    fc_velo = np.gradient(fc_traj, axis=1)
    fo_velo = np.gradient(fo_traj, axis=1)

    # standardize between 0.1 and 1.1 for the machine learning algorithm (zeros will be ignored!)
    fc_velo = preprocessing.minmax_scale(fc_velo, feature_range=(0.1, 1.1), axis=1)
    fo_velo = preprocessing.minmax_scale(fo_velo, feature_range=(0.1, 1.1), axis=1)

    # both 'fc_velo' and 'fo_velo' should be in the shape (num_features, num_frames) (e.g. (12, 500) or (18, 500))
    # for the prediction we need the shape of (num_samples, num_frames, num_features)
    # num_samples = 1, num_frames = length of trial (e.g. 500), num_features = velocity of trajectories (e.g. 12 or 18)
    # check with rs_fc_velo.shape
    rs_fc_velo = reshape_data(fc_velo)
    rs_fo_velo = reshape_data(fo_velo)


    fc_model = tf.keras.models.load_model(r'models\FC_velo_model.h5')
    fc_prediction = fc_model(rs_fc_velo, training=False)

    fc_preds = fc_prediction.numpy()[0]

    [loc, height] = find_peaks(fc_preds[:, 1], height=MIN_PEAK_THRESHOLD, distance=25)

    l_heel = fc_traj[0,:]
    r_heel = fc_traj[4,:]

    for fc in loc:
        if l_heel[fc] < r_heel[fc]:
            vicon.CreateAnEvent(subject_name[0], "Left", "Foot Strike", int(fc + start_frame), 0.0)
        else:
            vicon.CreateAnEvent(subject_name[0], "Right", "Foot Strike", int(fc + start_frame), 0.0)


    fo_model = tf.keras.models.load_model(r'models\FO_velo_model.h5')
    fo_prediction = fo_model(rs_fo_velo, training=False)
    fc_preds = fo_prediction.numpy()[0]
    [loc, height] = find_peaks(fc_preds[:, 1], height=MIN_PEAK_THRESHOLD, distance=25)
    for fo in loc:
        if l_heel[fo] > r_heel[fo]:
            vicon.CreateAnEvent(subject_name[0], "Left", "Foot Off", int(fo + start_frame), 0.0)
        else:
            vicon.CreateAnEvent(subject_name[0], "Right", "Foot Off", int(fo + start_frame), 0.0)


    vicon.Disconnect()