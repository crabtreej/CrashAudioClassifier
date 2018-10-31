import sys
import parseXMLData
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

    print('Processed all audio data')
    sf = 32000

    # 100 milliseconds * sample frequency
    window_size = 0.1 * sf
    # 50% overlap, so 50 milliseconds * sample frequency for step size
    step_size = 0.05 * sf

    # it's a map that maps classID to a list of audio clips, those audio clips have been decomposed into a list of
    # frames, and each of those frames are represented by a list of thirteen MFCCs, so it's a list of lists of lists
    classToClipsAsFramesOfMFCCsMapA = getClassIDsToClipFramesMFCCs(classToClipsMapA)
    classToClipsAsFramesOfMFCCsMapB = getClassIDsToClipFramesMFCCs(classToClipsMapB)
    classToClipsAsFramesOfMFCCsMapC = getClassIDsToClipFramesMFCCs(classToClipsMapC)
    MFCCsMapList = []
    MFCCsMapList.append(classToClipsAsFramesOfMFCCsMapA)
    MFCCsMapList.append(classToClipsAsFramesOfMFCCsMapB)

    comboList = []
    for MFCCsMap in MFCCsMapList:
        for classID in MFCCsMap:
            #there are a certain amount of clips keyed to each class
            for clip in MFCCsMap[classID]:
                #a clip is an array of frames
                for frame in clip:
                    #a frame is an array of 13 mfccs
                    comboList.append(frame)

    # KMeans clustering of combo list
    kmeans = KMeans(n_clusters=64).fit(comboList)

    # KMeans predict on each clip now, get predicted label
    histograms = []
    classMembership = []
    for classID in classToClipsAsFramesOfMFCCsMapC:
        for clip in classToClipsAsFramesOfMFCCsMapC[classID]:
            predictedLabels = [0] * 128 
            prediction = kmeans.predict(clip)
            for frame in prediction:
                predictedLabels[frame] += 1
            histograms.append(predictedLabels)
            classMembership.append(classID)

    print(histograms)
    print(classMembership)   
            
    #now we have training data for an svm
    mySVM = SVC(gamma='auto')
    mySVM.fit(histograms[::2], classMembership[::2])
    print(mySVM.score(histograms[1::2], classMembership[1::2]))












