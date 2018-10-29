import sys
import parseXMLData
import convertToFrames


pathToData = '../Dataset/crashAudio/audio/'


if __name__ == '__main__':

    classToClipsMapA = parseXMLData.parseXMLCutWavs(pathToData + 'A' + '/')
    classToClipsMapB = parseXMLData.parseXMLCutWavs(pathToData + 'B' + '/')
    classToClipsMapC = parseXMLData.parseXMLCutWavs(pathToData + 'C' + '/')

    classToClipsA2 = convertToFrames.convertToFrames(classToClipsMapA)
    classToClipsB2 = convertToFrames.convertToFrames(classToClipsMapB)
    print('Processed all audio data')
