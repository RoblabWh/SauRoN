from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import QMainWindow
import RobotRepresentation, PickUpStation, DeliveryStation


class SimulationWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Simulation")
        self.width = 1000
        self.height = 600
        self.setGeometry(200, 100, self.width, self.height)
        self.setFixedWidth(self.width)
        self.setFixedHeight(self.height)

        self.robotRepresentation = 0
        self.pickUpStation = 0
        self.deliveryStation = 0

    def paintEvent(self, event):
        painter = QPainter(self)

        painter.begin(self)
        if self.pickUpStation != 0:
            self.pickUpStation.paint(painter)
        if self.deliveryStation != 0:
            self.deliveryStation.paint(painter)
        if self.robotRepresentation != 0:
            self.robotRepresentation.paint(painter)
        painter.end()

    def initRobot(self, robotStartPosX, robotStartPosY, robotStartDirection, robotWidth, robotLength):
        self.robotRepresentation = RobotRepresentation.RobotRepresentation(robotStartPosX,
                                                       robotStartPosY,
                                                       robotStartDirection,
                                                       robotWidth,
                                                       robotLength)

    def initPickUpStation(self, posX, posY, width, length):
        self.pickUpStation = PickUpStation.PickUpStation(posX, posY, width, length)

    def initDeliveryStation(self, posX, posY, width, length):
        self.deliveryStation = DeliveryStation.DeliveryStation(posX, posY, width, length)

    def updateRobot(self, posX, posY, direction):
        self.robotRepresentation.update(posX, posY, direction)
        self.repaint()
