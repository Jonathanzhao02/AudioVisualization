"""
Contains all implementations of Qt objects
"""

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets


class FramelessWindow(QtWidgets.QMainWindow):
    """
    Removes border of QMainWindow while adding in
    basic functionality such as resizing or closing
    """

    def __init__(self):
        """
        Sets up style and data after calling
        parent __init__ method
        """

        QtWidgets.QMainWindow.__init__(self)
        self.oldPos = self.pos()
        self.resizeGrip = None
        self.exitButton = None
        self.setWindowFlags(QtCore.Qt.WindowFlags(QtCore.Qt.FramelessWindowHint))

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
        self.exitButton = ExitButton(self)

    def resizeEvent(self, event):
        """
        Sets the geometry of the exit/resizing widgets
        in addition to resizing the window

        :param event: The resize event
        :type event: QEvent
        """

        self.exitButton.setGeometry(self.width() - 20, 0, 20, 20)
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
