import sys
import parseXMLData
import convertToFrames


pathToData = '../Dataset/crashAudio/audio/'


if __name__ == '__main__':

    classToClipsMapA = parseXMLData.parseXMLCutWavs(pathToData + 'A' + '/')
    classToClipsMapB = parseXMLData.parseXMLCutWavs(pathToData + 'B' + '/')
    classToClipsMapC = parseXMLData.parseXMLCutWavs(pathToData + 'C' + '/')

    print('Processed all audio data')
