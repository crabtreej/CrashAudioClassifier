import sys
import gc

def convertToFrames(classToClipsMap):
    classToClipsMap2 = []
    for k in classToClipsMap:
        print(k)
        segmentArray = []
        for c in classToClipsMap[k]:
            start = 0
            end = 0
            length = c.size
            samplingRate = 32000 / 10
            overLap = int(samplingRate * .50)
            end = int(end + samplingRate)
            frameArray = []

            while(end < length-1):
                index = start    
                for x in range(start, end):
                    frameArray.append(c[index])
                    index = index + 1
                start = end - overLap
                end = end + overLap
                if end >= length:
                    end = length - 1
                index = start 
                for x in range(start, end):
                    frameArray.append(c[index])
                    index = index + 1
                start = start + overLap
                end = end + overLap
            segmentArray.append(frameArray)
        classToClipsMap2.append(segmentArray)
    gc.collect()
    return classToClipsMap2

