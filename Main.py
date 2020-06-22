import pyaudio

from AudioVisualizer import AudioVisualizer

PyAudio = pyaudio.PyAudio()
devices = [PyAudio.get_device_info_by_index(i) for i in range(PyAudio.get_device_count())]

FORMAT = pyaudio.paInt16
CHANNELS = min(PyAudio.get_default_input_device_info()['maxInputChannels'], PyAudio.get_default_output_device_info()['maxOutputChannels'], 1)
RATE = int(min(PyAudio.get_default_input_device_info()['defaultSampleRate'], PyAudio.get_default_output_device_info()['defaultSampleRate']))
FRAMES_PER_BUFFER = 1024
LOG_MODE = False

visualizer = AudioVisualizer(PyAudio=PyAudio, format=FORMAT, channels=CHANNELS, sample_rate=RATE, chunk_size=FRAMES_PER_BUFFER, log_mode=LOG_MODE)
visualizer.start()