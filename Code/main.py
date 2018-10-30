import sys
import parseXMLData
from sklearn.cluster import KMeans
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
#import sounddevice as sd
import time
from pyAudioAnalysis import audioFeatureExtraction as fe

pathToData = '../Dataset/crashAudio/audio/'


def getClassIDsToMFCCs(classIDsToClipsMap):
    classToMFCCsOfClips = {}
    for classID in classIDsToClipsMap:
        classToMFCCsOfClips[classID] = []
        for singleEventAudioClip in classIDsToClipsMap[classID]:
            #this returns a matrix that looks like:
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
            #so featuresMatrix[0] = a list of the zcr for each window, and featuresMatrix[0][0] 
            #would be the zcr for the very first window
            #we care about featuresMatrix[8:20] for the 13 MFCCs
            featuresMatrix, featureNames = fe.stFeatureExtraction(singleEventAudioClip, sf, window_size, step_size)
            mfccsForEachWindow = []
            #for each window, loop through indicies 8-20 to get all of its mfccs into one list, and compile all of
            #that into a bigger list
            for i in range(0, len(featuresMatrix[8])):
                mfccsForOneWindow = []
                for j in range(8, 21):
                    mfccsForOneWindow.append(featuresMatrix[j][i])
                mfccsForEachWindow.append(mfccsForOneWindow)

            #have list that looks like [ [mfcc0, mfcc1, mfcc2...], [mfcc0...], ... ] for each window in one clip
            classToMFCCsOfClips[classID].append(mfccsForEachWindow)


if __name__ == '__main__':

    classToClipsMapA = parseXMLData.parseXMLCutWavs(pathToData + 'A' + '/')
    classToClipsMapB = parseXMLData.parseXMLCutWavs(pathToData + 'B' + '/')
    classToClipsMapC = parseXMLData.parseXMLCutWavs(pathToData + 'C' + '/')

    print('Processed all audio data')
    sf = 32000

    #100 milliseconds * sample frequency
    window_size = 0.1 * sf
    #50% overlap, so 50 milliseconds * sample frequency for step size
    step_size = 0.05 * sf


    #it's a map that maps classID to a list of audio clips, those audio clips have been decomposed into a list of
    #frames, and each of those frames are represented by a list of thirteen MFCCs, so it's a list of lists of lists
    classToMFCCsMapA = getClassIDsToMFCCs(classToClipsMapA)
    classToMFCCsMapB = getClassIDsToMFCCs(classToClipsMapB)
    classToMFCCsMapC = getClassIDsToMFCCs(classToClipsMapC)
    
    print(classToMFCCsMapA)
    print(classToMFCCsMapB)
    print(classToMFCCsMapC)
    
    comboList = []
    for classID in classToMFCCsMapA:
        for frame in classID:
            for MFCC in frame:
                comboList.append[MFCC]

    #KMeans clustering of combo list
    kmeans = KMeans(n_clusters=64).fit(comboList)

    #KMeans predict on each MFCC now, get predicted label 
    predictedLabels = []
    for classID in classToMFCCsMapA:
        for frame in classID:
            for MFCC in frame:
                prediction = kmeans.predict(MFCC) #returns 0-63
                predictedLabels[prediction] += 1 #increment index at index 

    #Now we have an array with predicted clusters for each MFCC
    
