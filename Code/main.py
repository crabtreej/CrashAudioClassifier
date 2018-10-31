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

def getHistogramsAndMembershipFromKMeans(kmeansCLF, classToClipsMFCCsMap):
    histograms = []
    classMembership = []
    for classID in classToClipsMFCCsMap:
        for clip in classToClipsMFCCsMap[classID]:
            predictedLabels = [0] * 64
            prediction = kmeans.predict(clip)
            for frame in prediction:
                predictedLabels[frame] += 1
            histograms.append(predictedLabels)
            classMembership.append(classID)

    return (histograms, classMembership)

if __name__ == '__main__':

    with open("classToClipsAsFramesOfMFCCsMapA.txt", "rb") as fp:
        classToClipsAsFramesOfMFCCsMapA = pickle.load(fp)
    with open("classToClipsAsFramesOfMFCCsMapB.txt", "rb") as fp:
        classToClipsAsFramesOfMFCCsMapB = pickle.load(fp)
    with open("classToClipsAsFramesOfMFCCsMapC.txt", "rb") as fp:
        classToClipsAsFramesOfMFCCsMapC = pickle.load(fp)
    
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
    trainingHistograms, trainingLabels = getHistogramsAndMembershipFromKMeans(kmeans, classToClipsAsFramesOfMFCCsMapA)
    tempHist, tempLabels = (getHistogramsAndMembershipFromKMeans(kmeans, classToClipsAsFramesOfMFCCsMapB))
    trainingHistograms.extend(tempHist)
    trainingLabels.extend(tempLabels)

    validationHistograms, validationLabels = getHistogramsAndMembershipFromKMeans(kmeans, classToClipsAsFramesOfMFCCsMapC)

    print(trainingHistograms)
    print(trainingLabels)

    #now we have training data for an svm
    mySVM = SVC(gamma='auto')
    mySVM.fit(trainingHistograms, trainingLabels)
    print(mySVM.score(validationHistograms, validationLabels))












