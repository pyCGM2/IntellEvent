"""
The module contains procedures for detecting foot contact event.

check out the script : *\Tests\test_events.py* for examples
"""

from typing import List, Tuple, Dict, Optional,Union
import pyCGM2; LOGGER = pyCGM2.LOGGER

import numpy as np
import pandas as pd
import btk
from sklearn import preprocessing
from scipy.signal import find_peaks


from pyCGM2.Events.eventProcedures import EventProcedure
from pyCGM2.Tools import btkTools

import intellevent



#-------- EVENT PROCEDURES  ----------


class IntellEventProcedure(EventProcedure):
    """
    Gait event detection procedure based on Intellevent

    """

    def __init__(self,globalFrame,forwardProgression,
                 server_mode=True,
                 min_peak_threshold = 0.2, distance = 25):
        """
        Initializes the IntellEventProcedure class.
        """
        super(IntellEventProcedure, self).__init__()
        self.description = "intellevent"

        self.m_globalFrameOrientation = globalFrame
        self.m_forwardProgression = forwardProgression

        self.m_markers = ["LHEE", "LTOE", "LANK", "RHEE", "RTOE", "RANK"]

        self.m_min_peak_threshold = min_peak_threshold
        self.m_distance = distance
        self.m_serverMode = server_mode
        
        self._baseFrequency= 100

    def __prepare(self,acq):

        def reshape_data(traj):
            rs_traj = np.transpose(np.array(traj).reshape(1, traj.shape[0], traj.shape[1]), (0, 2, 1))
            return rs_traj
        
        def resample_data(traj, sample_frequ, frequ_to_sample):
            period = '{}ns'.format(int(1e9 / sample_frequ))
            index = pd.date_range(0, periods=len(traj[0, :]), freq=period)
            resampled_data = [pd.DataFrame(val, index=index).resample('{}ns'.format(int(1e9 / frequ_to_sample))).mean() for val in traj]
            resampled_data = [np.array(traj.interpolate(method='linear')) for traj in resampled_data]
            resampled_data = np.concatenate(resampled_data, axis=1)
            return resampled_data
        
        
        n_markers = len(self.m_markers)

        if self.m_globalFrameOrientation == "XYZ":
            if self.m_forwardProgression:
               Rglobal= np.array([[1, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 1]])
            else:
                Rglobal= np.array([[-1, 0, 0],
                                  [0, -1, 0],
                                  [0, 0, 1]])

        if self.m_globalFrameOrientation == "YXZ":
            if self.m_forwardProgression:
                Rglobal= np.array([[0, -1, 0],
                                  [1, 0, 0],
                                  [0, 0, 1]])
            else:
                Rglobal= np.array([[0, 1, 0],
                                  [-1, 0, 0],
                                  [0, 0, 1]])

        for marker in self.m_markers:
            point = acq.GetPoint(marker)
            values = np.zeros((acq.GetPointFrameNumber(),3))
            for i in range (0, acq.GetPointFrameNumber()):
                values[i,:] = np.dot(Rglobal.T,point.GetValues()[i,:])
            point.SetValues(values)

        array = btkTools.markersToArray(acq,self.m_markers,gathercolumn=True)
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


        #Down / up sampling?
        cam_frequency = acq.GetPointFrequency()
        if cam_frequency != self._baseFrequency:
            rs_fc_velo = resample_data(fc_velo, cam_frequency, self._baseFrequency).transpose()
            rs_fo_velo = resample_data(fo_velo, cam_frequency, self._baseFrequency).transpose()
        else:
            rs_fc_velo = fc_velo
            rs_fo_velo = fo_velo


        # both 'fc_velo' and 'fo_velo' should be in the shape (num_features, num_frames) (e.g. (12, 500) or (18, 500))
        # for the prediction we need the shape of (num_samples, num_frames, num_features)
        # num_samples = 1, num_frames = length of trial (e.g. 500), num_features = velocity of trajectories (e.g. 12 or 18)
        # check with rs_fc_velo.shape
        rs_fc_velo = reshape_data(fc_velo)
        rs_fo_velo = reshape_data(fo_velo)

        LOGGER.logger.info("[intellevent] - prepare data----> done")
        return rs_fc_velo, rs_fo_velo


    def __predict(self,fc_data,fo_data,server_mode=True ):

        if self.m_serverMode:
            try:
                import requests

                fc_preds = requests.post('http://127.0.0.1:5000/predict_fc', json={'traj': fc_data.tolist()})
                fc_preds = np.array(fc_preds.json())[0]

                fo_preds = requests.post('http://127.0.0.1:5000/predict_fo', json={'traj': fo_data.tolist()})
                fo_preds = np.array(fo_preds.json())[0]

                LOGGER.logger.info("[intellevent] - predict data----> done")
                return fc_preds, fo_preds
            except Exception as e:
               LOGGER.logger.error( "[intellevent error ( server mode )] [%s] " %( str(e)) )


        if not self.m_serverMode:
            try:
                import tensorflow as tf
                fc_model = tf.keras.models.load_model(intellevent.DATA_TRAINED_MODEL_PATH+ "FC_velo_model.h5")
                fc_prediction = fc_model(fc_data, training=False)
                fc_preds = fc_prediction.numpy()[0]

                fo_model = tf.keras.models.load_model(intellevent.DATA_TRAINED_MODEL_PATH+ "FO_velo_model.h5")
                fo_prediction = fc_model(fo_data, training=False)
                fo_preds = fo_prediction.numpy()[0]

                LOGGER.logger.info("[intellevent] - predict data----> done")
                return fc_preds, fo_preds
            except Exception as e:
               LOGGER.logger.error( "[intellevent error ( server mode : not connected )] [%s] " %( str(e)) )



    def detect(self,acq:btk.btkAcquisition)-> Union[Tuple[int, int, int, int], int] :
        """
        Detect events using the intellevent.

        Args:
            acq (btk.btkAcquisition): A BTK acquisition instance containing motion capture data.

        Returns:
            Union[Tuple[int, int, int, int], int]: Frames indicating the left foot strike, left foot off, 
                                                   right foot strike, and right foot off respectively. 
                                                   Returns 0 if detection fails.
        """


        ff = acq.GetFirstFrame()
        lhee = acq.GetPoint("LHEE").GetValues()
        rhee = acq.GetPoint("RHEE").GetValues()

        LOGGER.logger.info("[intellevent] - prepare data")
        fc_data,fo_data = self.__prepare(btk.btkAcquisition.Clone(acq))
        LOGGER.logger.info("[intellevent] - predict data")
        fc_preds,fo_preds = self.__predict(fc_data,fo_data)
        
        indexes_fs_left=[]
        indexes_fo_left=[]
        indexes_fs_right =[]
        indexes_fo_right=[]

        [loc, height] = find_peaks(fc_preds[:, 1], height=self.m_min_peak_threshold, 
                                   distance=self.m_distance)
        
        loc = np.ceil( (loc / self._baseFrequency) * acq.GetPointFrequency())

        loc = [int(it) for it in loc]
        
        if self.m_globalFrameOrientation[0]== "X":axis = 0
        if self.m_globalFrameOrientation[0]== "Y":axis = 1

        for index in loc:
            
            if lhee[index,axis] < rhee[index,axis]:
                indexes_fs_left.append(index+ff)
            else:
                indexes_fs_right.append(index+ff)

        [loc, height] = find_peaks(fo_preds[:, 1], 
                                   height=self.m_min_peak_threshold, 
                                   distance=self.m_distance)
        loc = np.ceil( (loc / self._baseFrequency) * acq.GetPointFrequency())
        loc = [int(it) for it in loc]

        for index in loc:
            if lhee[index,axis] > rhee[index,axis]:
                indexes_fo_left.append(index+ff)
            else:
                indexes_fo_right.append(index+ff)
        
        return indexes_fs_left,indexes_fo_left, indexes_fs_right, indexes_fo_right        
    