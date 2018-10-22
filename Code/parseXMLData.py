import xml.etree.ElementTree as ET
import os
from scipy.io import wavfile

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
    
    wavsAndClassIDs = []
    

def parseXMLCutWavs(pathToXMLFiles):
    for fName in os.listdir(pathToXMLFiles):
        if fName.endswith('.xml'):
            print(f'Currently Parsing XML file: {fName}...')
            #list of tuples of (classID, startTime, endTime)
            timesAndClassIDs = getClassIDsAndTimes(os.path.join(pathToXMLFiles, fName))

            #grab the actual wav file by dropping the .xml on the file name
            wavFileName = os.path.splitext(fName)[0] + '.wav'
            print(f'Got time stamp and class membership info. Loading {wavFileName}...')
       
            #read the .wav and cut it into separate arrays based on the info
            loadAndCutWav(os.path.join(pathToXMLFiles, wavFolder, wavFileName), timesAndClassIDs)
            #put that in a map of classID -> set of arrays of .wav info

                
