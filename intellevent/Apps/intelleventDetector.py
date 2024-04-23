
import argparse
import sys
import os

import pyCGM2
LOGGER = pyCGM2.LOGGER

from pyCGM2.Events import eventFilters
from pyCGM2.Lib.Processing import progression

# pkgPath = "C:\\Users\\fleboeuf\\Documents\\Programmation\\git folders\\intellevent(fork)"
# if pkgPath not in sys.path: sys.path.append(pkgPath)

import intellevent
from intellevent.procedure import eventProcedure

def main(args=None):

    if args  is None:
        parser = argparse.ArgumentParser(description='Intelle event gait event Detector')
        args = parser.parse_args()

    try:
        from viconnexusapi import ViconNexus
        NEXUS = ViconNexus.ViconNexus()
        from pyCGM2.Nexus import nexusFilters
        from pyCGM2.Nexus import nexusTools
        NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()
    except:
        LOGGER.logger.error("Vicon nexus not connected")
        NEXUS_PYTHON_CONNECTED = False


    if NEXUS_PYTHON_CONNECTED: # run Operation



        # ----------------------INPUTS-------------------------------------------
        # --- acquisition file and path----
        DATA_PATH, reconstructFilenameLabelledNoExt = nexusTools.getTrialName(NEXUS)

        reconstructFilenameLabelled = reconstructFilenameLabelledNoExt+".c3d"

        LOGGER.logger.info("data Path: " + DATA_PATH)
        LOGGER.logger.info("calibration file: " + reconstructFilenameLabelled)

        #acqGait = btkTools.smartReader(str(DATA_PATH + reconstructFilenameLabelled))

    #     # --------------------------SUBJECT -----------------------------------

        # Notice : Work with ONE subject by session
        subject = nexusTools.getActiveSubject(NEXUS)
        LOGGER.logger.info("Subject name : " + subject)

        # --- btk acquisition ----
        nacf = nexusFilters.NexusConstructAcquisitionFilter(NEXUS,
            DATA_PATH, reconstructFilenameLabelledNoExt, subject)
        acqGait = nacf.build()

        # acqGait =  btkTools.applyTranslators(acqGait,translators)

        progressionAxis, forwardProgression, globalFrame =progression.detectProgressionFrame(acqGait)

        # event procedure
        evp = eventProcedure.IntellEventProcedure(globalFrame,forwardProgression)

        # event filter
        evf = eventFilters.EventFilter(evp, acqGait)
        evf.detect()

        # # ----------------------DISPLAY ON VICON-------------------------------
        nexusTools.createEvents(NEXUS, subject, acqGait, [ "Foot Strike", "Foot Off"])
        # ========END of the nexus OPERATION if run from Nexus  =========

        NEXUS.Disconnect()

    else:
        return 0


if __name__ == "__main__":
    main(args=None)
