#Load the required libraries:
#   * scipy
#   * numpy
#   * matplotlib
from scipy.io import wavfile
from matplotlib import pyplot as plt
import numpy as np

# Load the data and calculate the time of each sample
samplerate, data = wavfile.read('../Dataset/crashAudio/audio/A/v2/00001_1.wav')
data = data[5*samplerate:int(5.5*samplerate)]
data = data / max(data)
times = (np.arange(0, len(data), 1)/float(samplerate))*1000

# Make the plot
# You can tweak the figsize (width, height) in inches
plt.figure(figsize=(10, 4))
plt.fill_between(times, data)
plt.xlim(times[0], times[-1])
plt.xlabel('time (ms)')
plt.ylabel('amplitude')
# You can set the format by changing the extension
# like .pdf, .svg, .eps
plt.savefig('plot.png', dpi=100)
plt.show()
