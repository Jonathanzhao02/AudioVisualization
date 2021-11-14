# AudioVisualization
Creates a PyQt application to visualize the frequency spectrum and waveform of incoming computer audio.

## Built With
* [PyQt5](https://www.riverbankcomputing.com/software/pyqt/)
* [NumPy](https://numpy.org/)
* [SciPy](https://www.scipy.org/)
* [PyAudio](http://people.csail.mit.edu/hubert/pyaudio/)

## Usage
To use the visualizer, simply import and initialize the AudioVisualizer object with a PyAudio stream.

```python
"""
Sets up and executes an instance of an AudioVisualizer object using computer audio
"""

import pyaudio

from audio_visualizer import AudioVisualizer

py_audio = pyaudio.PyAudio()

FORMAT = pyaudio.paInt16
CHANNELS = min(py_audio.get_default_input_device_info()['maxInputChannels'],
               py_audio.get_default_output_device_info()['maxOutputChannels'],
               2)
RATE = int(min(py_audio.get_default_input_device_info()['defaultSampleRate'],
               py_audio.get_default_output_device_info()['defaultSampleRate']))
FRAMES_PER_BUFFER = 1024

visualizer = AudioVisualizer(py_audio=py_audio,
                             data_format=FORMAT,
                             channels=CHANNELS,
                             sample_rate=RATE,
                             chunk_size=FRAMES_PER_BUFFER,
                             wav_reflect=True)
visualizer.start()
```
