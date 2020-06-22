import numpy as np
import math
import sys
import wave
import pyaudio

from queue import Queue
from threading import Thread, Event

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from PyQt5.QtWidgets import QApplication

from scipy.signal.windows import blackmanharris


class AudioVisualizer:
    def __init__(self, PyAudio, format=pyaudio.paInt16, channels=1, sample_rate=48000, chunk_size=1024, log_mode=False,
                 decay_speed=0.5):
        # setting object variables
        self.PyAudio = PyAudio
        self.format = format
        self.channels = channels
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.log_mode = log_mode
        self.decay_speed = decay_speed

        self.fft_size = int(self.chunk_size / 2)
        self.fft_size = 2 ** 8

        self.max_freq = int(self.sample_rate / 2)
        self.min_freq = 0

        # x-axis data for plotting
        self.x = np.arange(0, 2 * self.chunk_size)

        # log mode does not support step mode, so x-axis must be shortened
        if self.log_mode:
            self.x_fft = np.linspace(0, self.max_freq, self.fft_size)
        else:
            self.x_fft = np.linspace(0, self.max_freq, self.fft_size + 1)

        # sets up QtPy application + window
        pg.setConfigOptions(antialias=True)
        self.traces = dict()
        self.app = QApplication(sys.argv)
        self.win = pg.GraphicsWindow()
        self.win.setGeometry(5, 115, 1560, 1070)

        # creates both plots
        self.waveform = self.win.addPlot(
            row=1, col=1,
        )

        self.spectrum = self.win.addPlot(
            row=1, col=1
        )

        # hides plot axes
        self.waveform.hideAxis('bottom')
        self.waveform.hideAxis('left')
        self.spectrum.hideAxis('bottom')
        self.spectrum.hideAxis('left')

        # additional variables
        self.frames = []
        self.queue = Queue(-1)
        self.prev_y_fft = None
        self.prev_y = None

    def set_plotdata(self, name, data_x, data_y):

        # runs if data has been plotted before
        if name in self.traces:
            self.traces[name].setData(data_x, data_y)

        # runs if data has not been plotted yet
        else:

            # sets up waveform plot
            if name == 'waveform':
                self.traces[name] = self.waveform.plot(pen='c', width=10)
                self.waveform.setYRange(0, 1, padding=0)
                self.waveform.setXRange(0, self.chunk_size * 2, padding=0.005)

            # sets up frequency spectrum plot
            if name == 'spectrum':
                self.spectrum.setLogMode(x=self.log_mode, )
                self.spectrum.setYRange(0, 1, padding=0)

                if self.log_mode:
                    self.traces[name] = self.spectrum.plot(symbol='o')
                    self.spectrum.setXRange(
                        np.log10(20), np.log10(self.max_freq), padding=0.005)
                else:
                    grad = QtGui.QLinearGradient(0, 0, 0, 1)
                    grad.setColorAt(0, pg.mkColor('r'))
                    grad.setColorAt(0.5, pg.mkColor('#ffa500'))
                    grad.setColorAt(1, pg.mkColor('y'))
                    grad.setCoordinateMode(QtGui.QGradient.ObjectMode)
                    brush = QtGui.QBrush(grad)
                    self.traces[name] = self.spectrum.plot(data_x, data_y, pen=None, width=10,
                                                           fillLevel=0, fillBrush=brush,
                                                           stepMode=True)
                    self.spectrum.setXRange(
                        20, self.max_freq, padding=0.005)

    # ripped and edited from Stack Overflow
    def decode(self, in_data, channels, format=np.int16):
        """
        Convert a byte stream into a 2D numpy array with
        shape (chunk_size, channels)

        Samples are interleaved, so for a stereo stream with left channel
        of [L0, L1, L2, ...] and right channel of [R0, R1, R2, ...], the output
        is ordered as [L0, R0, L1, R1, ...]
        """

        if channels > 2:
            print("Only up to 2 channels supported!")

        # read data from buffer as specified format (we default to paInt16 so the stream should be read as int16)
        result = np.frombuffer(in_data, dtype=format)

        # calculate length of data after splitting into channels
        chunk_length = len(result) / channels
        chunk_length = int(chunk_length)

        # reshape data into L/R channel if 2 channels, otherwise data stays the same
        result = np.reshape(result, (chunk_length, channels))

        return result

    def update(self):
        # get and pre-process data
        data = self.queue.get()
        self.queue.queue.clear()
        data = self.decode(data, self.channels, np.int16)

        # calculate waveform of data
        y = np.mean(data, axis=1)
        y = y / self.max_freq
        y = (y + 1) / 2

        # smooths waveform values to be more easy on the eyes
        if self.prev_y is not None:
            y = (1 - self.decay_speed) * self.prev_y + self.decay_speed * y

        # new variable to store any non-essential y transformations
        new_y = np.concatenate((y, np.flipud(y)))

        # apply blackman harris window to data
        window = blackmanharris(self.chunk_size)
        data = window * np.mean(data, axis=1)

        # calculate fourier transform of data
        y_fft = np.abs(np.fft.rfft(data, n=self.fft_size * 2))
        y_fft = np.delete(y_fft, len(y_fft) - 1)
        y_fft = y_fft * 2 / (self.max_freq * 256)
        y_fft = y_fft ** 0.5

        # smooths frequency spectrum values to be more easy on the eyes
        if self.prev_y_fft is not None:
            y_fft = (1 - self.decay_speed) * self.prev_y_fft + self.decay_speed * y_fft

        # plot all data
        self.set_plotdata('waveform', self.x, new_y)
        self.set_plotdata('spectrum', self.x_fft, y_fft)

        # previous value updates
        self.prev_y = y
        self.prev_y_fft = y_fft

    def open_stream(self):
        # creates stream to computer's audio devices, aka audio I/O
        stream = self.PyAudio.open(format=self.format,
                                   channels=self.channels,
                                   rate=self.sample_rate,
                                   input=True,
                                   frames_per_buffer=self.chunk_size, )

        # reads from stream and adds to queue until stopped
        while not self.event.is_set():
            data = stream.read(self.chunk_size)
            self.queue.put(data)
            self.frames.append(data)

        # closes all streams
        stream.close()
        self.PyAudio.terminate()

    def start(self):
        # creates PyQt timer to automatically update the graph every 20ms
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(20)

        # sets up stream thread
        self.event = Event()
        self.thread = Thread(target=self.open_stream)
        self.thread.start()

        # runs the application if not already running
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QApplication.instance().exec_()

        # stops input thread
        self.event.set()

        # writes recorded audio to file
        """wf = wave.open("louden.wav", 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.PyAudio.get_sample_size(self.format))
        wf.setframerate(self.sample_rate)
        wf.writeframes(b''.join(self.frames))
        wf.close()"""
