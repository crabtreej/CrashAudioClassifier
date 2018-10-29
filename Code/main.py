import sys
import parseXMLData
import convertToFrames
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import time

pathToData = '../Dataset/crashAudio/audio/'

if __name__ == '__main__':

    classToClipsMapA = parseXMLData.parseXMLCutWavs(pathToData + 'A' + '/')
    classToClipsMapB = parseXMLData.parseXMLCutWavs(pathToData + 'B' + '/')
    classToClipsMapC = parseXMLData.parseXMLCutWavs(pathToData + 'C' + '/')

    print('Processed all audio data')

    sf = 32000.

    #trying to plot the power vs. hz to make sure it makes sense
    # Define window length (4 seconds)
    win = 1 * 32000
    freqs, psd = signal.welch(classToClipsMapA[2][0], 32000, nperseg=1024, scaling='density')

    # Plot the power spectrum
    #sns.set(font_scale=1.2, style='white')
    plt.figure(figsize=(8, 4))
    plt.plot(freqs, psd, color='k', lw=2)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power spectral density (V^2 / Hz)')
    plt.ylim([0, psd.max() * 1.1])
    plt.title("Welch's periodogram")
    plt.xlim([0, 20000])
    plt.show()
    #sns.despine()
