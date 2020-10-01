from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import QMainWindow
import RobotRepresentation
from Station import Station


def initRobots(robots, scaleFactor, mode):

    robotRepresentations = []
    for robot in robots:
        robot_draw = RobotRepresentation.RobotRepresentation(robot.getPosX(),
                                                             robot.getPosY(),
                                                             robot.getDirectionAngle(),
                                                             robot.width,
                                                             robot.length,
                                                             scaleFactor,
                                                             mode)
        robotRepresentations.append(robot_draw)
    return robotRepresentations


def initStations(stations, scaleFactor):
    _stations = []
    for i, station in enumerate(stations):
        station_draw = Station(station.posX, station.posY, station.width, station.length, i, scaleFactor)
        _stations.append(station_draw)
    return _stations


class SimulationWindow(QMainWindow):

    def __init__(self, application, robots, stations, args):
        super().__init__()

        self.app = application
        self.setWindowTitle("Simulation")
        self.width = int(args.arena_width*args.scale_factor)
        self.height = int(args.arena_length*args.scale_factor)
        self.setGeometry(200, 100, self.width, self.height)
        self.setFixedWidth(self.width)
        self.setFixedHeight(self.height)

        self.robotRepresentations = initRobots(robots, args.scale_factor, args.mode)
        self.stations = stations#initStations(stations, args.scale_factor)

        self.painter = QPainter(self)

    def paintEvent(self, event):

        self.painter.begin(self)
        for station in self.stations:
            station.paint(self.painter)
        for robot in self.robotRepresentations:
            robot.paint(self.painter)
        self.painter.end()

    def updateRobot(self, robot, num):
        self.robotRepresentations[num].update(robot.getPosX(), robot.getPosY(), robot.getDirectionAngle(), robot.radarHits)
        self.repaint()
        self.app.processEvents()
