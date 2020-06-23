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
    and draw the audio's waveform and frequency spectrum
    """

    def __init__(self, py_audio, data_format=pyaudio.paInt16,
                 channels=1, sample_rate=48000, chunk_size=1024,
                 bass_frequency=160, low_frequency=0, high_frequency=20000,
                 log_mode=False, wav_decay_speed=0.5, fft_decay_speed=0.5, bass_decay_speed=1,
                 width=800, height=800):
        # setting object variables
        self.py_audio = py_audio
        self.data_format = data_format
        self.channels = channels
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.log_mode = log_mode
        self.wav_decay_speed = wav_decay_speed
        self.fft_decay_speed = fft_decay_speed
        self.bass_decay_speed = bass_decay_speed

        # calculating other important values
        self.fft_size = int(self.chunk_size / 2)

        self.max_freq = int(self.sample_rate / 2)
        self.min_freq = 0

        if self.max_freq < high_frequency:
            high_frequency = self.max_freq

        self.bass_index = int(bass_frequency / self.max_freq * self.fft_size)
        self.low_index = int(low_frequency / self.max_freq * self.fft_size)
        self.high_index = int(high_frequency / self.max_freq * self.fft_size)

        # sets up QtPy application
        pg.setConfigOptions(antialias=True)
        self.traces = dict()
        self.app = QApplication(sys.argv)
        self.win = QtWidgets.QMainWindow()

        # dimension-related variables
        self.width = width
        self.height = height

        self.center_x = self.width / 2
        self.center_y = self.height / 2
        self.center_offset = min(self.width, self.height) / 4
        self.radius = min(self.width, self.height) / 4
        self.max_offset = self.radius / 4

        # QtPy graphic objects
        self.label = QtWidgets.QLabel()
        self.canvas = QtGui.QPixmap(self.width, self.height)
        self.label.setPixmap(self.canvas)
        self.win.setCentralWidget(self.label)

        self.painter = None
        self.cpen = QtGui.QPen(QtCore.Qt.cyan)
        self.gpen = QtGui.QPen()
        self.black = pg.mkColor('#000000')

        # additional variables
        self.frames = []
        self.queue = Queue(-1)
        self.event = Event()

        self.prev_y_fft = None
        self.prev_y_wav = None
        self.prev_bass = None

    @staticmethod
    def intermediate(val, low, high, val_low, val_high):
        """"
        Maps a value to another range

        :param val: The value to be mapped
        :type val: float

        :param low: The low limit of the map range
        :type low: float

        :param high: The high limit of the map range
        :type high: float

        :param val_low: The low limit of the value range
        :type val_low: float

        :param val_high: The high limit of the value range
        :type val_high: float
        """
        return (val - val_low) / (val_high - val_low) * (high - low) + low

    def get_gradient_pen(self, val, delta):
        """
        Returns a pen containing a color from the gradient

        :param val: The intermediate value on the gradient [0, 1]
        :type val: float

        :param delta: The alpha of the color and relative width of the pen
        :type val: float
        """
        r1, g1, b1 = 254, 254, 0

        r2, g2, b2 = 102, 225, 250

        r3, g3, b3 = 254, 0, 157

        bound1 = 1/2

        bound2 = 9/11

        if val < bound1:
            r, g, b = self.intermediate(val % 1, r1, r2, 0, bound1),\
                      self.intermediate(val % 1, g1, g2, 0, bound1),\
                      self.intermediate(val % 1, b1, b2, 0, bound1)
        elif val < bound2:
            r, g, b = self.intermediate(val % 1, r2, r3, bound1, bound2),\
                      self.intermediate(val % 1, g2, g3, bound1, bound2),\
                      self.intermediate(val % 1, b2, b3, bound1, bound2)
        else:
            r, g, b = self.intermediate(val % 1, r3, r1, bound2, 1),\
                      self.intermediate(val % 1, g3, g1, bound2, 1),\
                      self.intermediate(val % 1, b3, b1, bound2, 1)

        color = QtGui.QColor(min(r + self.intermediate(lighten, 0, 255 - r, 0, 1), 255),
                             min(g + self.intermediate(lighten, 0, 255 - g, 0, 1), 255),
                             min(b + self.intermediate(lighten, 0, 255 - b, 0, 1), 255))
        pen = QtGui.QPen(color)
        pen.setWidth(self.intermediate(lighten, 1, 3, 0, 1))
        return pen

    def draw_data(self, y_fft, y, val=1):
        """
        Draws data onto canvas

        :param y_fft: The fourier transform data
        :type y_fft: numpy array

        :param y: The waveform data
        :type y: numpy array

        :param val: The amount to expand the ring
        :type val: float
        """
        self.painter = QtGui.QPainter(self.label.pixmap())
        self.painter.fillRect(0, 0, self.width, self.height, self.black)

        offset = self.max_offset * val

        angle = np.linspace(-np.pi * 3 / 2, np.pi / 2, len(y_fft)) * -1

        center_x = self.center_x + np.cos(angle) * (self.center_offset + offset)
        center_y = self.center_y + np.sin(angle) * (self.center_offset + offset)

        rot_x = y_fft * np.cos(angle)
        rot_y = y_fft * np.sin(angle)

        rot_x, rot_y = rot_x * (self.radius + offset), rot_y * (self.radius + offset)

        rot_x *= 4 - val
        rot_y *= 4 - val

        for i in np.arange(len(y_fft)):
            self.painter.setPen(self.get_gradient_pen(i / len(y_fft), val))
            self.painter.drawLine(QtCore.QLineF(int(center_x[i]), int(center_y[i]),
                                      int(center_x[i] + rot_x[i]), int(center_y[i] + rot_y[i])))

        x_vals = np.linspace(0, self.width, self.chunk_size * 2)
        y_vals = y * self.center_y + self.center_y

        points = QtGui.QPolygonF()

        for i in np.arange(self.chunk_size * 2):
            points.append(QtCore.QPointF(int(x_vals[i]), int(y_vals[i])))

        self.painter.setPen(self.cpen)
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
        data = self.decode(data, self.channels, np.int16)

        # calculate waveform of data
        y_wav = np.mean(data, axis=1)
        y_wav = y_wav / self.max_freq   # shifts y_wav to [-1, 1]

        # smooths waveform values to be more easy on the eyes
        if self.prev_y_wav is not None:
            y_wav = (1 - self.wav_decay_speed) * self.prev_y_wav + self.wav_decay_speed * y_wav

        # new variable to store any non-essential y_wav transformations
        new_y = np.concatenate((y_wav, np.flipud(y_wav)))

        # apply blackman harris window to data
        window = blackmanharris(self.chunk_size)
        data = window * np.mean(data, axis=1)

        # calculate fourier transform of data
        y_fft = np.abs(np.fft.rfft(data, n=self.fft_size * 2))
        y_fft = np.delete(y_fft, len(y_fft) - 1)
        y_fft = y_fft * 2 / (self.max_freq * 256)   # shifts y_fft to [0, 1]
        y_fft = y_fft ** 0.7                        # emphasizes smaller peaks

        # smooths frequency spectrum values to be more easy on the eyes
        if self.prev_y_fft is not None:
            y_fft = (1 - self.fft_decay_speed) * self.prev_y_fft + self.fft_decay_speed * y_fft

        # calculates average values of bass frequencies
        bass = np.mean(y_fft[0:int(self.bass_index)])

        # smooths bass values
        if self.prev_bass is not None:
            bass = (1 - self.bass_decay_speed) * self.prev_bass + self.bass_decay_speed * bass

        # draws data
        self.draw_data(y_fft[self.low_index:self.high_index], new_y, bass ** 2)

        # previous value updates
        self.prev_y_wav = y_wav
        self.prev_y_fft = y_fft
        self.prev_bass = bass

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

            if not self.queue.empty():
                self.queue.queue.clear()    # ensures the application always reads the newest data

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