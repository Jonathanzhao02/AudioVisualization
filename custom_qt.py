from pyqtgraph.Qt import QtCore, QtGui, QtWidgets


class FramelessWindow(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.oldPos = self.pos()
        self.resizeGrip = None
        self.exitButton = None
        self.setWindowFlags(QtCore.Qt.WindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint))

    def addWidgets(self, centralWidget):
        self.setCentralWidget(centralWidget)
        self.resizeGrip = ResizeGrip(self)
        self.exitButton = ExitButton(self)

    def resizeEvent(self, event):
        self.exitButton.setGeometry(self.width() - 20, 0, 20, 20)
        self.resizeGrip.setGeometry(0, 0, 20, 20)
        QtWidgets.QMainWindow.resizeEvent(self, event)

    def changeEvent(self, event):

        if self.resizeGrip is not None:
            self.resizeGrip.resizing = False

    def mousePressEvent(self, event):
        self.oldPos = event.globalPos()
        self.resizeGrip.resizing = False

    def mouseMoveEvent(self, event):

        if self.resizeGrip is not None and not self.resizeGrip.resizing:
            delta = QtCore.QPoint(event.globalPos() - self.oldPos)
            self.move(self.x() + delta.x(), self.y() + delta.y())
            self.oldPos = event.globalPos()


class ResizeGrip(QtWidgets.QSizeGrip):
    def __init__(self, parent):
        QtWidgets.QSizeGrip.__init__(self, parent)
        self.resizing = False
        self.setPalette(QtGui.QPalette(QtGui.QColor('transparent')))

    def mousePressEvent(self, event):
        QtWidgets.QSizeGrip.mousePressEvent(self, event)
        self.resizing = True


class ExitButton(QtWidgets.QLabel):
    def __init__(self, parent):
        QtWidgets.QLabel.__init__(self, parent)
        self.setPalette(QtGui.QPalette(QtGui.QColor('transparent')))
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
        before calling the parent QLabel __init__ method

        :param visualizer: The corresponding AudioVisualizer
        :type visualizer: AudioVisualizer
        """

        self.visualizer = visualizer
        QtWidgets.QLabel.__init__(self)

    def resizeEvent(self, event):
        """
        Override of the resizeEvent method that also updates
        the QLabel QPixmap and visualizer object dimensions

        :param event: The resize event
        :type event: QEvent
        """

        pixmap = self.pixmap()
        self.setPixmap(pixmap.scaled(self.width(), self.height(), aspectRatioMode=QtCore.Qt.IgnoreAspectRatio))
        self.visualizer.set_dims(self.width(), self.height())
        QtWidgets.QLabel.resizeEvent(self, event)
