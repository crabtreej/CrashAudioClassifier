import sys
import parseXMLData

pathToData = '../Dataset/crashAudio/audio/'




if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Please input the 3 folds you want to use as test data.')
        exit(0)

    for foldName in sys.argv[1:]:
        classToClipsMap = parseXMLData.parseXMLCutWavs(pathToData + foldName + '/')
