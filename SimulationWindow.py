from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import QMainWindow
import RobotRepresentation
from Station import Station


class SimulationWindow(QMainWindow):

    def __init__(self, application):
        super().__init__()

        self.app = application
        self.setWindowTitle("Simulation")
        self.width = 1000
        self.height = 600
        self.setGeometry(200, 100, self.width, self.height)
        self.setFixedWidth(self.width)
        self.setFixedHeight(self.height)

        self.robotRepresentation = 0
        self.pickUpStation = 0
        self.deliveryStation = 0
        self.painter = QPainter(self)


    def paintEvent(self, event):

        self.painter.begin(self)
        if self.pickUpStation != 0:
            self.pickUpStation.paint(self.painter)
        if self.deliveryStation != 0:
            self.deliveryStation.paint(self.painter)
        if self.robotRepresentation != 0:
            self.robotRepresentation.paint(self.painter)
        self.painter.end()

    def initRobot(self, robotStartPosX, robotStartPosY, robotStartDirection, robotWidth, robotLength):
        self.robotRepresentation = RobotRepresentation.RobotRepresentation(robotStartPosX,
                                                       robotStartPosY,
                                                       robotStartDirection,
                                                       robotWidth,
                                                       robotLength)

    def initPickUpStation(self, posX, posY, width, length):
        self.pickUpStation = Station(posX, posY, width, length, 0)

    def initDeliveryStation(self, posX, posY, width, length):
        self.deliveryStation = Station(posX, posY, width, length, 1)

    def updateRobot(self, posX, posY, direction):
        self.robotRepresentation.update(posX, posY, direction)
        self.repaint()
        self.app.processEvents()
