"""
Contains the cooler AudioVisualizer object
"""

import sys
import time

from queue import Queue
from threading import Thread, Event

import pyaudio
import numpy as np

import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication

from scipy.signal.windows import blackmanharris, tukey

from custom_qt import *


class AudioVisualizer:
    """
    Takes a py_audio instance to create an audio stream
    and draw the audio's waveform and frequency spectrum
    """

    def __init__(self, py_audio, data_format=pyaudio.paInt16,
                 channels=1, sample_rate=48000, chunk_size=1024,
                 bass_frequency=260, low_frequency=0, high_frequency=20000,
                 wav_decay_speed=0.5, fft_decay_speed=0.5, bass_decay_speed=0.8,
                 wav_amp_factor=1, fft_amp_factor=0.7, bass_amp_factor=0.8,
                 bass_max_amp=3,
                 tukey_alpha=0.04, width=800, height=800,
                 wav_reflect=False, fft_reflect=False, fft_symmetrical=False):
        """
        Initializes necessary variables and the QApplication objects

        :param py_audio: The PyAudio instance to be used
        :type py_audio: PyAudio

        :param data_format: The format of the PyAudio stream data
        :type data_format: int

        :param channels: The number of channels in the audio stream
        :type channels: int

        :param sample_rate: The sample rate of the audio stream
        :type sample_rate: int

        :param chunk_size: The number of bytes per audio stream read
        :type chunk_size: int

        :param bass_frequency: The estimated bass frequency of the audio to use for the bass visual effect
        :type bass_frequency: int

        :param low_frequency: The lowest frequency to display
        :type low_frequency: int

        :param high_frequency: The highest frequency to display
        :type high_frequency: int

        :param wav_decay_speed: The rate at which the waveform should decay
        :type wav_decay_speed: float

        :param fft_decay_speed: The rate at which the fourier transform should decay
        :type fft_decay_speed: float

        :param bass_decay_speed: The rate at which the bass visual effect should decay
        :type bass_decay_speed: float

        :param wav_amp_factor: The exponent to apply to the waveform (lower = higher amplitude)
        :type wav_amp_factor: float

        :param fft_amp_factor: The exponent to apply to the fourier transform (lower = higher peaks)
        :type fft_amp_factor: float

        :param bass_amp_factor: The exponent to apply to the bass visual effect (lower = more sensitive trigger)
        :type bass_amp_factor: float

        :param bass_max_amp: The maximum amp size of the bass vfx (lower = smaller difference in bar height at max bass)
        :type bass_max_amp: float

        :param tukey_alpha: The alpha of the tukey window applied to the fourier transform (higher = lower low Hz peaks)
        :type tukey_alpha: float

        :param width: The initial width of the window
        :type width: int

        :param height: The initial height of the window
        :type height: int

        :param wav_reflect: Whether the waveform should be reflected across the center of the window
        :type wav_reflect: bool

        :param fft_reflect: Whether the fourier transform should be reflected on the inside of the circle
        :type fft_reflect: bool

        :param fft_symmetrical: Whether the fourier transform should be reflected on the other side of the circle
        :type fft_reflect: bool
        """

        # setting object variables
        self.py_audio = py_audio
        self.data_format = data_format
        self.channels = channels
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.wav_decay_speed = wav_decay_speed
        self.fft_decay_speed = fft_decay_speed
        self.bass_decay_speed = bass_decay_speed
        self.wav_amp_factor = wav_amp_factor
        self.fft_amp_factor = fft_amp_factor
        self.bass_amp_factor = bass_amp_factor
        self.bass_max_amp = bass_max_amp
        self.tukey_alpha = tukey_alpha
        self.wav_reflect = wav_reflect
        self.fft_reflect = fft_reflect
        self.fft_symmetrical = fft_symmetrical

        # calculating other important values
        self.fft_size = self.chunk_size / 2

        # helps to save frame rate by drawing half as much data
        if self.fft_symmetrical:
            self.fft_size /= 2

        self.fft_size = int(self.fft_size)

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
        self.win = FramelessWindow()

        # dimension-related variables
        self.width = width
        self.height = height

        self.center_x = self.width / 2
        self.center_y = self.height / 2
        self.center_radius = min(self.width, self.height) / 4
        self.fft_radius = min(self.width, self.height) / 8
        self.max_offset = self.fft_radius / 4

        # QtPy graphic objects
        self.label = Canvas(self)
        self.label.setMinimumSize(100, 100)
        self.canvas = QtGui.QPixmap(self.width, self.height)
        self.label.setPixmap(self.canvas)

        self.win.addWidgets(self.label)
        self.win.setWindowTitle("Spectrum")

        self.painter = None
        self.cpen = pg.mkPen('c')

        # additional variables
        self.frames = []
        self.queue = Queue(-1)
        self.event = Event()

        self.prev_y_fft = None
        self.prev_y_wav = None
        self.prev_bass = None

        self.frame_count = 0
        self.start_time = time.time()

    def set_dims(self, width, height):
        """
        Sets the dimensions of the visualizer and recalculates necessary dimensions

        :param width: New width of window
        :type width: int

        :param height: New height of window
        :type height: int
        """

        self.width = width
        self.height = height

        self.center_x = self.width / 2
        self.center_y = self.height / 2
        self.center_radius = min(self.width, self.height) / 4
        self.fft_radius = min(self.width, self.height) / 8
        self.max_offset = self.fft_radius / 4

    @staticmethod
    def intermediate(val, low, high, val_low=0, val_high=1):
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

        :return: Returns a value mapped between the low and high bounds
        """

        if low == high:
            return high

        if val_low == val_high:
            return high

        return (val - val_low) / (val_high - val_low) * (high - low) + low

    def get_gradient_pen(self, val, delta):
        """
        Returns a pen containing a color from the gradient

        :param val: The intermediate value on the gradient [0, 1]
        :type val: float

        :param delta: The alpha of the color and relative width of the pen
        :type val: float

        :return: Returns a QPen with a color from the gradient
        """

        bound1 = 1/2
        bound2 = 9/11

        r1, g1, b1 = 255, 255, 0
        r2, g2, b2 = 102, 225, 250
        r3, g3, b3 = 255, 0, 157

        if val < bound1:
            r, g, b = self.intermediate(val, r1, r2, 0, bound1), \
                      self.intermediate(val, g1, g2, 0, bound1), \
                      self.intermediate(val, b1, b2, 0, bound1)
        elif val < bound2:
            r, g, b = self.intermediate(val, r2, r3, bound1, bound2), \
                      self.intermediate(val, g2, g3, bound1, bound2), \
                      self.intermediate(val, b2, b3, bound1, bound2)
        else:
            r, g, b = self.intermediate(val, r3, r1, bound2, 1), \
                      self.intermediate(val, g3, g1, bound2, 1), \
                      self.intermediate(val, b3, b1, bound2, 1)

        color = QtGui.QColor(int(r),
                             int(g),
                             int(b),)
        color = color.lighter(int(100 + delta * 100))
        pen = QtGui.QPen(color)
        pen.setWidth(int(self.intermediate(delta, 1, 3)))
        return pen

    def draw_data(self, y_wav, y_fft, val):
        """
        Draws data onto the QLabel

        :param y_wav: The waveform data
        :type y_wav: ndarray

        :param y_fft: The fourier transform data
        :type y_fft: ndarray

        :param val: The amount to expand the ring [0, 1]
        :type val: float
        """

        # clears screen
        self.painter = QtGui.QPainter(self.label.pixmap())
        self.painter.fillRect(0, 0, self.width, self.height, QtCore.Qt.black)

        # calculates the waveform coordinates
        if self.wav_reflect:
            y_wav = np.concatenate((y_wav, np.flipud(y_wav)))

        x_vals = np.linspace(0, self.width, len(y_wav))
        y_vals = y_wav * self.center_y + self.center_y

        # creates points to be drawn
        points = QtGui.QPolygonF()

        for i in np.arange(len(y_wav)):
            points.append(QtCore.QPointF(int(x_vals[i]), int(y_vals[i])))

        # draws the points on to the QPixmap with a cyan pen
        self.painter.setPen(self.cpen)
        self.painter.drawPoints(points)

        # calculates coordinates for the frequency spectrum circle
        if not self.fft_symmetrical:
            y_fft[0:int(len(y_fft) / 2)] *= tukey(len(y_fft), alpha=self.tukey_alpha)[0:int(len(y_fft) / 2)]

        if self.fft_symmetrical:
            y_fft = np.concatenate((y_fft, np.flipud(y_fft)))

        offset = self.max_offset * val

        angle = np.linspace(-np.pi * 3 / 2, np.pi / 2, len(y_fft)) * -1

        center_x = self.center_x + np.cos(angle) * (self.center_radius + offset)
        center_y = self.center_y + np.sin(angle) * (self.center_radius + offset)

        rot_x = y_fft * np.cos(angle)
        rot_y = y_fft * np.sin(angle)

        rot_x = rot_x * (self.fft_radius + offset)
        rot_y = rot_y * (self.fft_radius + offset)

        rot_x *= 1 + val * self.bass_max_amp
        rot_y *= 1 + val * self.bass_max_amp

        # draws the lines on to the QPixmap with a color gradient pen
        if self.fft_reflect:
            for i in np.arange(len(y_fft)):
                self.painter.setPen(self.get_gradient_pen(i / len(y_fft), val))
                self.painter.drawLine(QtCore.QLineF(
                                      int(center_x[i] - rot_x[i]),
                                      int(center_y[i] - rot_y[i]),
                                      int(center_x[i] + rot_x[i]),
                                      int(center_y[i] + rot_y[i])))
        else:
            for i in np.arange(len(y_fft)):
                self.painter.setPen(self.get_gradient_pen(i / len(y_fft), val))
                self.painter.drawLine(QtCore.QLineF(
                                      int(center_x[i]),
                                      int(center_y[i]),
                                      int(center_x[i] + rot_x[i]),
                                      int(center_y[i] + rot_y[i])))

        # updates the window graphics
        self.painter.end()
        self.win.update()

    # ripped and edited from Stack Overflow
    @staticmethod
    def decode(in_data, channels, data_format=np.int16):
        """
        Convert a byte stream into a 2D numpy array with
        shape (chunk_size, channels)

        Samples are interleaved, so for a stereo stream with left channel
        of [L0, L1, L2, ...] and right channel of [R0, R1, R2, ...], the output
        is ordered as [L0, R0, L1, R1, ...]
        
        :param in_data: The byte array to split into channels
        :type in_data: list
        
        :param channels: The number of channels in the audio
        :type channels: int
        
        :param data_format: The data format of the bytes
        :type data_format: type

        :return: Returns an ndarray with the channels split into separate indices
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

        # apply blackman harris window to data
        window = blackmanharris(self.chunk_size)
        data = window * np.mean(data, axis=1)

        # calculate fourier transform of data
        y_fft = np.abs(np.fft.rfft(data, n=self.fft_size * 2))
        y_fft = np.delete(y_fft, len(y_fft) - 1)
        y_fft = y_fft * 2 / (self.max_freq * 256)   # shifts y_fft to [0, 1]

        # calculates average values of bass frequencies
        bass = np.max(y_fft[0:self.bass_index])

        # smooths bass values
        if self.prev_bass is not None:
            bass = (1 - self.bass_decay_speed) * self.prev_bass + self.bass_decay_speed * bass

        # smooths frequency spectrum values to be more easy on the eyes
        if self.prev_y_fft is not None:
            y_fft = (1 - self.fft_decay_speed) * self.prev_y_fft + self.fft_decay_speed * y_fft

        # draws data
        self.draw_data(y_wav ** self.wav_amp_factor,
                       y_fft[self.low_index:self.high_index] ** self.fft_amp_factor,
                       bass ** self.bass_amp_factor)

        # previous value updates
        self.prev_y_wav = y_wav
        self.prev_y_fft = y_fft
        self.prev_bass = bass

        self.frame_count += 1

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
        Executes the QApplication and PyAudio stream thread
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

        print("FPS", self.frame_count / (time.time() - self.start_time))
