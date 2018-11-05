import sys
import parseXMLData
import pickle
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
#import sounddevice as sd
import time
from pyAudioAnalysis import audioFeatureExtraction as fe

pathToData = '../Dataset/crashAudio/audio/'

def getClassIDsToClipFramesMFCCs(classIDsToClipsMap):
    classToMFCCsOfClips = {}
    for classID in classIDsToClipsMap:
        classToMFCCsOfClips[classID] = []
        for singleEventAudioClip in classIDsToClipsMap[classID]:
            # this returns a matrix that looks like:
            """
                  zcr   | energy | energy_entropy | .... | mfcc_0 | mfcc_1 | ... | ... |
            win0  value   value
            ----
            win1  value
            ----
            .
            .
            .
            """
            # so featuresMatrix[0] = a list of the zcr for each window, and featuresMatrix[0][0]
            # would be the zcr for the very first window
            # we care about featuresMatrix[8:20] for the 13 MFCCs

            featuresMatrix, featureNames = fe.stFeatureExtraction(
                singleEventAudioClip, sf, window_size, step_size)
            mfccsForEachWindow = []
            # for each window, loop through indicies 8-20 to get all of its mfccs into one list, and compile all of
            # that into a bigger list
            for i in range(0, len(featuresMatrix[8])):
                mfccsForOneWindow = []
                for j in range(8, 21):
                    mfccsForOneWindow.append(featuresMatrix[j][i])
                mfccsForEachWindow.append(mfccsForOneWindow)

            # have list that looks like [ [mfcc0, mfcc1, mfcc2...], [mfcc0...], ... ] for each window in one clip
            classToMFCCsOfClips[classID].append(mfccsForEachWindow)

    return classToMFCCsOfClips

if __name__ == '__main__':

    classToClipsMapA = parseXMLData.parseXMLCutWavs(pathToData + 'A' + '/')
    classToClipsMapB = parseXMLData.parseXMLCutWavs(pathToData + 'B' + '/')
    classToClipsMapC = parseXMLData.parseXMLCutWavs(pathToData + 'C' + '/')
    classToClipsMapD = parseXMLData.parseXMLCutWavs(pathToData + 'D' + '/')

    print('Processed all audio data')
    sf = 32000

    # 100 milliseconds * sample frequency
    window_size = 0.1 * sf
    window_sizes = [(.05 * sf), (.1 * sf), (.3 * sf)]
    # 50% overlap, so 50 milliseconds * sample frequency for step size
    step_size = 0.05 * sf

    # it's a map that maps classID to a list of audio clips, those audio clips have been decomposed into a list of
    # frames, and each of those frames are represented by a list of thirteen MFCCs, so it's a list of lists of lists
    
    suffixes = ["50m", "100m", "300m"]
    i = 0
    for w in window_sizes:
        window_size = w
        classToClipsAsFramesOfMFCCsMapA = getClassIDsToClipFramesMFCCs(classToClipsMapA)
        classToClipsAsFramesOfMFCCsMapB = getClassIDsToClipFramesMFCCs(classToClipsMapB)
        classToClipsAsFramesOfMFCCsMapC = getClassIDsToClipFramesMFCCs(classToClipsMapC)
        classToClipsAsFramesOfMFCCsMapD = getClassIDsToClipFramesMFCCs(classToClipsMapD)

        with open("classToClipsAsFramesOfMFCCsMapA_" + suffixes[i] + ".txt", "wb") as fp:
            pickle.dump(classToClipsAsFramesOfMFCCsMapA, fp)
        with open("classToClipsAsFramesOfMFCCsMapB_" + suffixes[i] + ".txt", "wb") as fp:
            pickle.dump(classToClipsAsFramesOfMFCCsMapB, fp)
        with open("classToClipsAsFramesOfMFCCsMapC_" + suffixes[i] + ".txt", "wb") as fp:
            pickle.dump(classToClipsAsFramesOfMFCCsMapC, fp)
        with open("classToClipsAsFramesOfMFCCsMapD_" + suffixes[i] + ".txt", "wb") as fp:
            pickle.dump(classToClipsAsFramesOfMFCCsMapD, fp)
        i = i + 1
        
