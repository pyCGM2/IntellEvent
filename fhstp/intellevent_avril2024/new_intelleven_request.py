# Note: tensorflow, numpy and pandas MUST have the described versions
# otherwise the mode cannot be opened and the code won´t work.
#import matplotlib.pyplot as plt
from viconnexusapi import ViconNexus
# cd C:\Program Files\Vicon\Nexus2.14\SDK\Win64\Python
#pip install ./viconnexusapi
from sklearn import preprocessing
import numpy as np #conda install numpy==1.18.5
#from ezc3d import c3d #conda install -c conda-forge ezc3d
from utils import reshape_data, fc_pred, fo_pred, get_trial_infos, resample_data
import time
import threading
import sys
import argparse


#import pandas as pd #conda install pandas==1.2.0

# marker names which are used for the algorithm
# adapt names to how they are stored in your .c3d file
marker_list = ["LHEE", "LTOE", "LANK", "RHEE", "RTOE", "RANK"]
base_frequency = 150



if __name__=='__main__':
    x_traj, y_traj, z_traj = [], [], []
    args_in = sys.argv

    # Get Vicon Nexus specific information from the viconnexus API
    vicon = ViconNexus.ViconNexus()
    vicon.ClearAllEvents()
    path, file_name = vicon.GetTrialName()
    subject_name = vicon.GetSubjectNames()[0]
    start_frame, end_frame = vicon.GetTrialRegionOfInterest()


    # Get all marker label names from the .c3d file
    #labels = c3d_trial['parameters']['POINT']['LABELS']['value']
    #labels = vicon.GetMarkerNames(subject_name)
    # Get the corresponding index for each marker name in 'marker_list'

    try:
        for marker in marker_list:
            if vicon.HasTrajectory(subject_name, marker):
                x, y, z, _ = vicon.GetTrajectory(subject_name, marker)
                x_traj.append(x[start_frame - 1:end_frame - 1])
                y_traj.append(y[start_frame - 1:end_frame - 1])
                z_traj.append(z[start_frame - 1:end_frame - 1])
            else:
                xyz, _ = vicon.GetModelOutput(subject_name, marker)
                x_traj.append(xyz[0][start_frame - 1:end_frame - 1])
                y_traj.append(xyz[1][start_frame - 1:end_frame - 1])
                z_traj.append(xyz[2][start_frame - 1:end_frame - 1])
    except:
        print(f"No Marker with the name: {marker}")



    # The current best model uses the x and z axis velocity for the IC model
    # and the x, y, and z axis velocity for the FO model
    fc_traj = np.concatenate([x_traj, z_traj])
    fo_traj = np.concatenate([x_traj, y_traj, z_traj])

    # x and y-coordinates need to be standardized depending on the starting direction,
    # z coordinates are always the same
    if any(fc_traj[0, 0:10] < 0):
        fc_traj[0:6, :] = (fc_traj[0:6, :] - np.mean(fc_traj[0:6, :], axis=1).reshape(6,1)) * (-1)
        fo_traj[0:12, :] = (fo_traj[0:12, :] - np.mean(fo_traj[0:12, :], axis=1).reshape(12, 1)) * (-1)

    # calculate the first derivative (= velocity) of the trajectories
    fc_velo = np.gradient(fc_traj, axis=1)
    fo_velo = np.gradient(fo_traj, axis=1)

    # standardize between 0.1 and 1.1 for the machine learning algorithm (zeros will be ignored!)
    fc_velo = preprocessing.minmax_scale(fc_velo, feature_range=(0.1, 1.1), axis=1)
    fo_velo = preprocessing.minmax_scale(fo_velo, feature_range=(0.1, 1.1), axis=1)


    #Down / up sampling?
    cam_frequency = vicon.GetFrameRate()
    if cam_frequency != base_frequency:
        rs_fc_velo = resample_data(fc_velo, cam_frequency, base_frequency).transpose()
        rs_fo_velo = resample_data(fo_velo, cam_frequency, base_frequency).transpose()
    else:
        rs_fc_velo = fc_velo
        rs_fo_velo = fo_velo

    # both 'fc_velo' and 'fo_velo' should be in the shape (num_features, num_frames) (e.g. (12, 500) or (18, 500))
    # for the prediction we need the shape of (num_samples, num_frames, num_features)
    # num_samples = 1, num_frames = length of trial (e.g. 500), num_features = velocity of trajectories (e.g. 12 or 18)
    # check with rs_fc_velo.shape
    rs_fc_velo = reshape_data(rs_fc_velo) #rs_fc_velo
    rs_fo_velo = reshape_data(rs_fo_velo) #rs_fo_velo


    # Multithreading to run both predictions at the same time
    # speeds up processing
    t1 = threading.Thread(target=fc_pred, args=(rs_fc_velo.tolist(), fc_traj[0:6], subject_name, start_frame, vicon, cam_frequency))
    t2 = threading.Thread(target=fo_pred, args=(rs_fo_velo.tolist(), fc_traj[0:6], subject_name, start_frame, vicon, cam_frequency))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    #print(time.time() - timer)
    vicon.Disconnect()

#plt.plot(fc_velo[7,:])
#plt.show()

#plt.plot(fc_traj[7,:])
#plt.show()
