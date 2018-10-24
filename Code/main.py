import sys
import parseXMLData

pathToData = '../Dataset/crashAudio/audio/'




if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Please input the 3 folds you want to use as test data.')
        exit(0)

    classToClipsMap = dict()
    for foldName in sys.argv[1:]:
        tempClassToClipsMap = parseXMLData.parseXMLCutWavs(pathToData + foldName + '/')
        for key in tempClassToClipsMap:
            if key in classToClipsMap:
                classToClipsMap[key].extend(tempClassToClipsMap[key])
            else:
                classToClipsMap[key] = tempClassToClipsMap[key]
  
    print('Processed all audio data')
    #print(f'Len for 1: {len(classToClipsMap[1])}\n Len for 2: {len(classToClipsMap[2])}\n Len for 3: {len(classToClipsMap[3])}')
    #print(classToClipsMap)
