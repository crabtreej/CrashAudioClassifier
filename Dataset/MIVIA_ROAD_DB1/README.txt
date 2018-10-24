MIVIA audio events data set for road surveillance applications
version: MIVIA_ROAD_DB1
date: 12/12/2014

CREATED BY:
Alessia Saggese, Nicola Strisciuglio, Mario Vento
University of Salerno (ITALY), Dept. of Information Eng., Electrical Eng. and Applied Math., MIVIA Lab

Contact person: 
Nicola Strisciuglio
email: nstrisciuglio@unisa.it

The data set
The MIVIA road audio events data set is composed of a total of 400 events for road surveillance applications, namely tire skidding and car crashes. The events are divided into 4 folds of 100 events each, in order to account for cross-validation experiments.

Description
The sounds have been registered with an Axis P8221Audio Module and an Axis T83 omnidirectional microphone for audio surveillance applications , sampled at 32000 Hz and quantized at 16 bits per PCM sample. The audio clips are distributed as WAV files.
Each fold contains a number of audio file of about 1 minute duration, in which a series of hazardous events is superimposed to a typical road background sound. Each audio file has a different background sound, so that several different real situations are simulated.

More details are reported on the web site http://mivia.unisa.it/datasets/audio-analysis/mivia-road-audio-events-data-set/

MATLAB Script
In order to parse the XML files in MATLAB, the xml_toolbox is available in the folder MATLAB. Please use the function xml_load in order to read a XML file and obtain as result a MATLAB struct with the related information.

