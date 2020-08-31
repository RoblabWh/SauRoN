from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import QMainWindow
import RobotRepresentation
from Station import Station


def initRobots(robots):
    robotRepresentations = []
    for robot in robots:
        robot_draw = RobotRepresentation.RobotRepresentation(robot.getPosX(),
                                                             robot.getPosY(),
                                                             robot.startDirection,
                                                             robot.width,
                                                             robot.length)
        robotRepresentations.append(robot_draw)
    return robotRepresentations


def initStations(stations):
    _stations = []
    for i, station in enumerate(stations):
        station_draw = Station(station.posX, station.posY, station.width, station.length, i)
        _stations.append(station_draw)
    return _stations


class SimulationWindow(QMainWindow):

    def __init__(self, application, robots, stations):
        super().__init__()

        self.app = application
        self.setWindowTitle("Simulation")
        self.width = 1000
        self.height = 600
        self.setGeometry(200, 100, self.width, self.height)
        self.setFixedWidth(self.width)
        self.setFixedHeight(self.height)

        self.robotRepresentations = initRobots(robots)
        self.stations = initStations(stations)

        self.painter = QPainter(self)

    def paintEvent(self, event):

        self.painter.begin(self)
        for station in self.stations:
            station.paint(self.painter)
        for robot in self.robotRepresentations:
            robot.paint(self.painter)
        self.painter.end()

    def updateRobot(self, robot, num):
        self.robotRepresentations[num].update(robot.getPosX(), robot.getPosY(), robot.getDirection())
        self.repaint()
        self.app.processEvents()
