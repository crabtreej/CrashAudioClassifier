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
    #so featuresMatrix[0] = a list of the zcr for each window, and featuresMatrix[0][0] would be the zcr for the very first window

    #we care about featuresMatrix[8:20] for the 13 MFCCs
    featuresMatrix, featureNames = fe.stFeatureExtraction(classToClipsMapA[2][0], sf, window_size, step_size)

