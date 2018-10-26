import sys
import parseXMLData

pathToData = '../Dataset/crashAudio/audio/'




if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Please input the 3 folds you want to use as test data.')
        exit(0)

    classToClipsMapA = parseXMLData.parseXMLCutWavs(pathToData + 'A' + '/')
    classToClipsMapB = parseXMLData.parseXMLCutWavs(pathToData + 'B' + '/')
    classToClipsMapC = parseXMLData.parseXMLCutWavs(pathToData + 'C' + '/')

    print('Processed all audio data')
    #print(f'Len for 1: {len(classToClipsMap[1])}\n Len for 2: {len(classToClipsMap[2])}\n Len for 3: {len(classToClipsMap[3])}')
    #print(classToClipsMap)
