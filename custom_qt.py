"""
Contains all implementations of Qt objects
"""

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets


class FramelessWindow(QtWidgets.QMainWindow):
    """
    Removes border of QMainWindow while adding in
    basic functionality such as resizing or closing
    """

    def __init__(self, visualizer):
        """
        Sets up style and data after calling
        parent __init__ method
        """

        QtWidgets.QMainWindow.__init__(self)
        self.visualizer = visualizer
        self.oldPos = self.pos()
        self.resizeGrip = None
        self.settingsButton = None
        self.exitButton = None
        self.setWindowFlags(QtCore.Qt.WindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint))

    def addWidgets(self, centralWidget):
        """
        Adds all necessary widgets to allow for
        exiting/resizing, along with a central
        widget

        :param centralWidget: The central widget of the window
        :type centralWidget: QtWidget
        """

        self.setCentralWidget(centralWidget)
        self.resizeGrip = ResizeGrip(self)
        self.settingsButton = SettingsButton(self)
        self.exitButton = ExitButton(self)

    def resizeEvent(self, event):
        """
        Sets the geometry of the exit/resizing widgets
        in addition to resizing the window

        :param event: The resize event
        :type event: QEvent
        """

        self.exitButton.setGeometry(self.width() - 20, 0, 20, 20)
        self.settingsButton.setGeometry(20, 0, 20, 20)
        self.resizeGrip.setGeometry(0, 0, 20, 20)
        QtWidgets.QMainWindow.resizeEvent(self, event)

    def mousePressEvent(self, event):
        """
        Tracks the old position of the mouse
        and toggles off the resizing flag

        :param event: The mouse event
        :type event: QEvent
        """

        self.oldPos = event.globalPos()
        self.resizeGrip.resizing = False

    def mouseMoveEvent(self, event):
        """
        Moves the window with the mouse

        :param event: The mouse event
        :type event: QEvent
        """

        if self.resizeGrip is not None and not self.resizeGrip.resizing:
            delta = QtCore.QPoint(event.globalPos() - self.oldPos)
            self.move(self.x() + delta.x(), self.y() + delta.y())
            self.oldPos = event.globalPos()


class SettingsPanel(QtWidgets.QWidget):
    """
    Input panel for all adjustable parameters
    """
    def assign_bass_freq(self, text):
        self.visualizer.bass_freq = int(text or '0')
        self.visualizer.bass_index = int(self.visualizer.bass_freq / self.visualizer.max_freq * self.visualizer.fft_size)
    
    def assign_low_freq(self, text):
        self.visualizer.low_freq = int(text or '0')
        self.visualizer.low_index = int(self.visualizer.low_freq / self.visualizer.max_freq * self.visualizer.fft_size)
    
    def assign_high_freq(self, text):
        self.visualizer.high_freq = int(text or '1')
        self.visualizer.high_index = int(self.visualizer.high_freq / self.visualizer.max_freq * self.visualizer.fft_size)
    
    def assign_max_freq(self, text):
        self.visualizer.max_freq = int(text or '1')
        self.visualizer.bass_index = int(self.visualizer.bass_freq / self.visualizer.max_freq * self.visualizer.fft_size)
        self.visualizer.low_index = int(self.visualizer.low_freq / self.visualizer.max_freq * self.visualizer.fft_size)
        self.visualizer.high_index = int(self.visualizer.high_freq / self.visualizer.max_freq * self.visualizer.fft_size)

    def assign_wav_decay(self, text):
        self.visualizer.wav_decay_speed = float(text or '0')
    
    def assign_fft_decay(self, text):
        self.visualizer.fft_decay_speed = float(text or '0')
    
    def assign_bass_decay(self, text):
        self.visualizer.bass_decay_speed = float(text or '0')
    
    def assign_wav_amp_factor(self, text):
        self.visualizer.wav_amp_factor = float(text or '0')
    
    def assign_fft_amp_factor(self, text):
        self.visualizer.fft_amp_factor = float(text or '0')
    
    def assign_bass_amp_factor(self, text):
        self.visualizer.bass_amp_factor = float(text or '0')
    
    def assign_overall_amp_factor(self, text):
        self.visualizer.overall_amp_factor = float(text or '0')
    
    def assign_bass_max_amp(self, text):
        self.visualizer.bass_max_amp = float(text or '0')
    
    def assign_wav_reflect(self, checked):
        self.visualizer.wav_reflect = checked
    
    def assign_fft_reflect(self, checked):
        self.visualizer.fft_reflect = checked
    
    def assign_fft_symmetrical(self, checked):
        self.visualizer.fft_symmetrical = checked

    def __init__(self, button):
        """
        Sets up style and data after calling
        parent __init__ method
        """
        QtWidgets.QWidget.__init__(self)
        self.setStyleSheet("background-color: black; color: white;")
        self.button = button
        self.visualizer = button.visualizer
        
        layout = QtWidgets.QFormLayout()
        
        bass_freq_field = QtWidgets.QLineEdit()
        bass_freq_field.setValidator(QtGui.QIntValidator(0, self.visualizer.sample_rate // 2))
        bass_freq_field.setText(str(self.visualizer.bass_freq))
        bass_freq_field.textChanged.connect(self.assign_bass_freq)
        layout.addRow("Bass Frequency", bass_freq_field)

        low_freq_field = QtWidgets.QLineEdit()
        low_freq_field.setValidator(QtGui.QIntValidator(0, self.visualizer.sample_rate // 2))
        low_freq_field.setText(str(self.visualizer.low_freq))
        low_freq_field.textChanged.connect(self.assign_low_freq)
        layout.addRow("Low Frequency", low_freq_field)

        high_freq_field = QtWidgets.QLineEdit()
        high_freq_field.setValidator(QtGui.QIntValidator(1, self.visualizer.sample_rate // 2))
        high_freq_field.setText(str(self.visualizer.high_freq))
        high_freq_field.textChanged.connect(self.assign_high_freq)
        layout.addRow("High Frequency", high_freq_field)
        
        max_freq_field = QtWidgets.QLineEdit()
        max_freq_field.setValidator(QtGui.QIntValidator(1, self.visualizer.sample_rate // 2))
        max_freq_field.setText(str(self.visualizer.max_freq))
        max_freq_field.textChanged.connect(self.assign_max_freq)
        layout.addRow("Max Frequency", max_freq_field)

        wav_decay_speed_field = QtWidgets.QLineEdit()
        wav_decay_speed_field.setValidator(QtGui.QDoubleValidator(0, 1, 2))
        wav_decay_speed_field.setText(str(self.visualizer.wav_decay_speed))
        wav_decay_speed_field.textChanged.connect(self.assign_wav_decay)
        layout.addRow("Wave Decay Speed", wav_decay_speed_field)

        fft_decay_speed_field = QtWidgets.QLineEdit()
        fft_decay_speed_field.setValidator(QtGui.QDoubleValidator(0, 1, 2))
        fft_decay_speed_field.setText(str(self.visualizer.fft_decay_speed))
        fft_decay_speed_field.textChanged.connect(self.assign_fft_decay)
        layout.addRow("FFT Decay Speed", fft_decay_speed_field)

        bass_decay_speed_field = QtWidgets.QLineEdit()
        bass_decay_speed_field.setValidator(QtGui.QDoubleValidator(0, 1, 2))
        bass_decay_speed_field.setText(str(self.visualizer.bass_decay_speed))
        bass_decay_speed_field.textChanged.connect(self.assign_bass_decay)
        layout.addRow("Bass Decay Speed", bass_decay_speed_field)

        wav_exp_field = QtWidgets.QLineEdit()
        wav_exp_field.setValidator(QtGui.QDoubleValidator(0, 1, 2))
        wav_exp_field.setText(str(self.visualizer.wav_amp_factor))
        wav_exp_field.textChanged.connect(self.assign_wav_amp_factor)
        layout.addRow("Wave Exponent", wav_exp_field)

        fft_exp_field = QtWidgets.QLineEdit()
        fft_exp_field.setValidator(QtGui.QDoubleValidator(0, 1, 2))
        fft_exp_field.setText(str(self.visualizer.fft_amp_factor))
        fft_exp_field.textChanged.connect(self.assign_fft_amp_factor)
        layout.addRow("FFT Exponent", fft_exp_field)

        bass_exp_field = QtWidgets.QLineEdit()
        bass_exp_field.setValidator(QtGui.QDoubleValidator(0, 1, 2))
        bass_exp_field.setText(str(self.visualizer.bass_amp_factor))
        bass_exp_field.textChanged.connect(self.assign_bass_amp_factor)
        layout.addRow("Bass Exponent", bass_exp_field)

        overall_amp_field = QtWidgets.QLineEdit()
        overall_amp_field.setValidator(QtGui.QDoubleValidator(0, 10, 2))
        overall_amp_field.setText(str(self.visualizer.overall_amp_factor))
        overall_amp_field.textChanged.connect(self.assign_overall_amp_factor)
        layout.addRow("Amp Factor", overall_amp_field)

        bass_max_amp_field = QtWidgets.QLineEdit()
        bass_max_amp_field.setValidator(QtGui.QDoubleValidator(0, 10, 2))
        bass_max_amp_field.setText(str(self.visualizer.bass_max_amp))
        bass_max_amp_field.textChanged.connect(self.assign_bass_max_amp)
        layout.addRow("Bass Max Amp", bass_max_amp_field)

        wav_reflect_box = QtWidgets.QCheckBox()
        wav_reflect_box.setChecked(self.visualizer.wav_reflect)
        wav_reflect_box.stateChanged.connect(self.assign_wav_reflect)
        layout.addRow("Wave Reflect", wav_reflect_box)

        fft_reflect_box = QtWidgets.QCheckBox()
        fft_reflect_box.setChecked(self.visualizer.fft_reflect)
        fft_reflect_box.stateChanged.connect(self.assign_fft_reflect)
        layout.addRow("FFT Reflect", fft_reflect_box)

        fft_symmetrical_box = QtWidgets.QCheckBox()
        fft_symmetrical_box.setChecked(self.visualizer.fft_symmetrical)
        fft_symmetrical_box.stateChanged.connect(self.assign_fft_symmetrical)
        layout.addRow("FFT Symmetry", fft_symmetrical_box)

        self.setLayout(layout)
    
    def closeEvent(self, event):
        QtWidgets.QWidget.closeEvent(self, event)
        self.button.win = None


class ResizeGrip(QtWidgets.QSizeGrip):
    """
    Overrides the mousePressEvent of the QSizeGrip
    to allow for tracking when the user is actively
    resizing the window
    """

    def __init__(self, parent):
        """
        Sets up widget style and data after calling
        parent __init__ function

        :param parent: The parent of the QtWidget
        :type parent: QtWidget
        """

        QtWidgets.QSizeGrip.__init__(self, parent)
        self.resizing = False
        self.setPalette(QtGui.QPalette(QtCore.Qt.blue))

    def mousePressEvent(self, event):
        """
        Overriden method that indicates the user is
        actively using this widget to resize the window

        :param event: The mouse event
        :type event: QEvent
        """

        QtWidgets.QSizeGrip.mousePressEvent(self, event)
        self.resizing = True


class SettingsButton(QtWidgets.QLabel):
    """
    QWidget that allows the user to open a settings panel
    """

    def __init__(self, parent):
        """
        Sets up widget style after calling parent __init__ function

        :param parent: The parent of the QtWidget
        :type parent: QtWidget
        """

        QtWidgets.QLabel.__init__(self, parent)
        self.visualizer = parent.visualizer
        self.win = None
        self.setPalette(QtGui.QPalette(QtCore.Qt.transparent))
        self.setText('\u2699')
        self.setAlignment(QtCore.Qt.AlignCenter)

    def mousePressEvent(self, event):
        QtWidgets.QLabel.mousePressEvent(self, event)

        if not self.win:
            self.win = SettingsPanel(self)
            self.win.show()


class ExitButton(QtWidgets.QLabel):
    """
    QWidget that allows the user to exit the QApplication
    """

    def __init__(self, parent):
        """
        Sets up widget style after calling parent __init__ function

        :param parent: The parent of the QtWidget
        :type parent: QtWidget
        """

        QtWidgets.QLabel.__init__(self, parent)
        self.setPalette(QtGui.QPalette(QtCore.Qt.transparent))
        self.setText('X')
        self.setAlignment(QtCore.Qt.AlignCenter)

    def mousePressEvent(self, event):
        QtCore.QCoreApplication.exit()


class Canvas(QtWidgets.QLabel):
    """
    Overrides the resizeEvent method of a QLabel
    to allow for QPixmap auto-scaling
    """

    def __init__(self, visualizer):
        """
        Keeps a reference to the AudioVisualizer object
        before calling the parent __init__ function

        :param visualizer: The corresponding AudioVisualizer
        :type visualizer: AudioVisualizer
        """

        self.visualizer = visualizer
        QtWidgets.QLabel.__init__(self)

    def resizeEvent(self, event):
        """
        Override of the resizeEvent method. Also updates
        the QLabel's pixmap and AudioVisualizer object dimensions

        :param event: The resize event
        :type event: QEvent
        """

        pixmap = self.pixmap()
        self.setPixmap(pixmap.scaled(self.width(), self.height(), aspectRatioMode=QtCore.Qt.IgnoreAspectRatio))
        self.visualizer.set_dims(self.width(), self.height())
        QtWidgets.QLabel.resizeEvent(self, event)
