import xml.etree.ElementTree as ET
import os
from scipy.io import wavfile

#TODO: make listening to the .wav files some kind of verbosity option for debugging
#import sounddevice as sd
#import time

wavFolder = 'v2/'
wavLenSeconds = 60
wavBitsEncoding = 16

def getClassIDsAndTimes(pathToFile):
    eventsRoot = ET.parse(pathToFile).getroot()[0]

    #extract the timestamp and classID info into a list, one tuple for each chunk
    clipTimesAndClasses = []

    lastEventEndSecond = 0
    for child in eventsRoot:
        #child is called 'item', item has children called CLASS_ID 
        #and CLASS_NAME, .find() finds the direct descendant with that name
        currentEventStart = float(child.find('STARTSECOND').text)
        currentEventEnd = float(child.find('ENDSECOND').text)
        classID = int(child.find('CLASS_ID').text)
        
        #last event end second to current event start is always background
        #noise, which is class ID 1
        clipTimesAndClasses.append((1, lastEventEndSecond, currentEventStart))
        clipTimesAndClasses.append((classID, currentEventStart, currentEventEnd))
        lastEventEndSecond = currentEventEnd

    #want last event to end of clip for background noise 
    clipTimesAndClasses.append((1, lastEventEndSecond, wavLenSeconds))
    
    #TODO add some kind of verbosity print 
    #print(clipTimesAndClasses)

    return clipTimesAndClasses


def loadAndCutWav(pathToWav, timesAndClassIDs):
    sampleRate, wavData = wavfile.read(pathToWav)
    print('File successfully loaded. Dividing .wav into events...')

    #https://stackoverflow.com/a/26716031
    #wavfile from scipy reads the data in as a numpy array, but the
    #wav isn't normalized between -1 and 1, so we divide by max int
    #allowed by the number of the .wavs bits of encoding, 16 bits in
    #this case
    wavData = wavData / (float(2 ** (wavBitsEncoding - 1)) + 1.0)
    
    classIDToWavChunks = dict() 
    for classID, startTime, endTime in timesAndClassIDs:
        tempWavAudio = wavData[int(startTime * sampleRate):int(endTime * sampleRate)]

        if classID in classIDToWavChunks:
            classIDToWavChunks[classID].append(tempWavAudio)
        else:
            classIDToWavChunks[classID] = [tempWavAudio]

        #TODO: debug option
        #print(f'Playing audio chunk with classID {classID} to make sure it\'s correct')
        #print(f'Start: {startTime} End: {endTime} SampleRate: {sampleRate}')
        #sd.play(tempWavAudio, sampleRate)
        #time.sleep(endTime - startTime)

    return classIDToWavChunks


def parseXMLCutWavs(pathToXMLFiles):
    classIDsToWavData = dict()
    for fName in os.listdir(pathToXMLFiles):
        if fName.endswith('.xml'):
            print(f'Currently Parsing XML file: {fName}...')
            #list of tuples of (classID, startTime, endTime)
            timesAndClassIDs = getClassIDsAndTimes(os.path.join(pathToXMLFiles, fName))

            #grab the actual wav file by dropping the .xml on the file name
            wavFileName = os.path.splitext(fName)[0] + '_1.wav'
            print(f'Got time stamp and class membership info. Loading {wavFileName}...')
       
            #read the .wav and cut it into separate arrays based on the info
            #dict of classID -> list of arrays of .wav Data
            tempIDsToWavData = loadAndCutWav(os.path.join(pathToXMLFiles, wavFolder, wavFileName), timesAndClassIDs)

            #take each dictionary returned by loadAndCutWav and condense them into a larger dictionar of the same form
            for key in tempIDsToWavData:
                if key in classIDsToWavData:
                    classIDsToWavData[key].extend(tempIDsToWavData[key])
                else:
                    classIDsToWavData[key] = tempIDsToWavData[key]
            print('Finished loading and cutting this .wav...')
        else: 
            print(f'Skipping non XML file {fName}...')

    return classIDsToWavData
