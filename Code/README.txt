Just some general info about the dataset.

The class IDs are 3 = crash
2 = skid
1 = background noise

I figured out that each folder (A, B, C, D) are a different fold, and that the background noises
are all unique. Basically, we should just cut the audio for each file between different events,
and non-event audio is unique background noise we can use as data for the background noise classifier.

Also, the path to the audio data from the root of the project is: Dataset/crashAudio/audio/, and then either A,
B, C, or D to get to a specific fold. The XML files are in there, named 00001_1.xml, 00002_1.xml, etc. The audio
files are inside v2/, and are named the same as the XML files, just 00001_1.wav, etc.
