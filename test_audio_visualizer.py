"""
Contains the cooler AudioVisualizer object
"""

import sys

from queue import Queue
from threading import Thread, Event

import pyaudio
import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication

from scipy.signal.windows import blackmanharris


class AudioVisualizer:
    """
    Takes a py_audio instance to create an audio stream
    and plot the audio's waveform and frequency spectrum
    """

    def __init__(self, py_audio, data_format=pyaudio.paInt16,
                 channels=1, sample_rate=48000, chunk_size=1024,
                 log_mode=False, decay_speed=0.5,
                 width=800, height=800):
        # setting object variables
        self.py_audio = py_audio
        self.data_format = data_format
        self.channels = channels
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.log_mode = log_mode
        self.decay_speed = decay_speed

        self.fft_size = int(self.chunk_size / 2)

        self.max_freq = int(self.sample_rate / 2)
        self.min_freq = 0

        # sets up QtPy application
        pg.setConfigOptions(antialias=True)
        self.traces = dict()
        self.app = QApplication(sys.argv)
        self.win = QtWidgets.QMainWindow()

        self.width = width
        self.height = height

        self.label = QtWidgets.QLabel()
        self.canvas = QtGui.QPixmap(self.width, self.height)
        self.label.setPixmap(self.canvas)
        self.win.setCentralWidget(self.label)

        self.painter = None
        self.cpen = pg.mkPen('c')
        self.black = pg.mkColor('#000000')

        self.center_x = self.width / 2
        self.center_y = self.height / 2
        self.center_offset = min(self.width, self.height) / 4
        self.radius = min(self.width, self.height) / 4

        # additional variables
        self.frames = []
        self.queue = Queue(-1)
        self.event = Event()

        self.prev_y_fft = None
        self.prev_y_wav = None

    def draw_data(self, y_fft, y):
        """
        Draws data onto canvas

        :param data_y: The y component of the data
        :type data_y: numpy array
        """
        self.painter = QtGui.QPainter(self.label.pixmap())
        self.painter.fillRect(0, 0, self.width, self.height, self.black)

        self.painter.setPen(self.cpen)

        angle = np.linspace(-np.pi * 3 / 2, np.pi / 2, len(y_fft)) * -1

        center_x = self.center_x + np.cos(angle) * self.center_offset
        center_y = self.center_y + np.sin(angle) * self.center_offset

        rot_x = y_fft * np.cos(angle)
        rot_y = y_fft * np.sin(angle)

        rot_x, rot_y = rot_x * self.radius, rot_y * self.radius

        lines = []

        for i in np.arange(len(y_fft)):
            lines.append(QtCore.QLineF(int(center_x[i]), int(center_y[i]),
                                      int(center_x[i] + rot_x[i]), int(center_y[i] + rot_y[i])))

        self.painter.drawLines(lines)

        x_vals = np.linspace(0, self.width, self.chunk_size * 2)
        y_vals = y * self.center_y + self.center_y

        points = QtGui.QPolygonF()

        for i in np.arange(self.chunk_size * 2):
            points.append(QtCore.QPointF(int(x_vals[i]), int(y_vals[i])))

        self.painter.drawPoints(points)

        self.painter.end()
        self.win.update()


    # ripped and edited from Stack Overflow
    def decode(self, in_data, channels, data_format=np.int16):
        """
        Convert a byte stream into a 2D numpy array with
        shape (chunk_size, channels)

        Samples are interleaved, so for a stereo stream with left channel
        of [L0, L1, L2, ...] and right channel of [R0, R1, R2, ...], the output
        is ordered as [L0, R0, L1, R1, ...]
        """

        if channels > 2:
            print("Only up to 2 channels supported!")

        # read data from buffer as specified format (default int16)
        result = np.frombuffer(in_data, dtype=data_format)

        # calculate length of data after splitting into channels
        chunk_length = len(result) / channels
        chunk_length = int(chunk_length)

        # reshape data into L/R channel if 2 channels, otherwise data stays the same
        result = np.reshape(result, (chunk_length, channels))

        return result

    def update(self):
        """
        Retrieve and process data from the queue to update the
        PyQt plots
        """

        # get and pre-process data
        data = self.queue.get()
        self.queue.queue.clear()  # clears queue to ensure always grabbing newer data
        data = self.decode(data, self.channels, np.int16)

        # calculate waveform of data
        y_wav = np.mean(data, axis=1)
        y_wav = y_wav / self.max_freq   # shifts y_wav to [-1, 1]

        # smooths waveform values to be more easy on the eyes
        if self.prev_y_wav is not None:
            y_wav = (1 - self.decay_speed) * self.prev_y_wav + self.decay_speed * y_wav

        # new variable to store any non-essential y_wav transformations
        new_y = np.concatenate((y_wav, np.flipud(y_wav)))

        # apply blackman harris window to data
        window = blackmanharris(self.chunk_size)
        data = window * np.mean(data, axis=1)

        # calculate fourier transform of data
        y_fft = np.abs(np.fft.rfft(data, n=self.fft_size * 2))
        y_fft = np.delete(y_fft, len(y_fft) - 1)
        y_fft = y_fft * 2 / (self.max_freq * 256)   # shifts y_fft to [0, 1]
        y_fft = y_fft ** 0.5                        # emphasize smaller peaks

        # smooths frequency spectrum values to be more easy on the eyes
        if self.prev_y_fft is not None:
            y_fft = (1 - self.decay_speed) * self.prev_y_fft + self.decay_speed * y_fft

        # draw data
        self.draw_data(y_fft, new_y)

        # previous value updates
        self.prev_y_wav = y_wav
        self.prev_y_fft = y_fft

    def open_stream(self):
        """
        Opens a stream from the py_audio instance and
        continuously puts data into a queue
        """

        # creates stream to computer's audio devices, aka audio I/O
        stream = self.py_audio.open(format=self.data_format,
                                    channels=self.channels,
                                    rate=self.sample_rate,
                                    input=True,
                                    frames_per_buffer=self.chunk_size,)

        # reads from stream and adds to queue until stopped
        while not self.event.is_set():
            data = stream.read(self.chunk_size)
            self.queue.put(data)
            self.frames.append(data)

        # closes all streams
        stream.close()
        self.py_audio.terminate()

    def start(self):
        """
        Executes the application and stream I/O thread
        """
        self.win.show()

        # creates PyQt timer to automatically update the graph every 20ms
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(20)

        # sets up stream thread
        thread = Thread(target=self.open_stream)
        thread.start()

        # runs the application if not already running
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QApplication.instance().exec_()

        # stops input thread
        self.event.set()
        self.painter.end()
