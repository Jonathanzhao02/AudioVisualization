"""
Sets up and executes an instance of an AudioVisualizer object using computer audio
"""

import pyaudio

from audio_visualizer import AudioVisualizer

py_audio = pyaudio.PyAudio()
devices = [py_audio.get_device_info_by_index(i) for i in range(py_audio.get_device_count())]

FORMAT = pyaudio.paInt16
CHANNELS = min(py_audio.get_default_input_device_info()['maxInputChannels'],
               py_audio.get_default_output_device_info()['maxOutputChannels'],
               2)
RATE = int(min(py_audio.get_default_input_device_info()['defaultSampleRate'],
               py_audio.get_default_output_device_info()['defaultSampleRate']))
FRAMES_PER_BUFFER = 1024
LOG_MODE = False

visualizer = AudioVisualizer(py_audio=py_audio,
                             data_format=FORMAT,
                             channels=CHANNELS,
                             sample_rate=RATE,
                             chunk_size=FRAMES_PER_BUFFER,
                             log_mode=LOG_MODE)
visualizer.start()
