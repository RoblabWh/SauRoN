from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import QMainWindow, QWidget, QPushButton, QLabel
import RobotRepresentation
from Station import Station


def initRobots(robots, scaleFactor, mode):

    robotRepresentations = []
    for i, robot in enumerate(robots):
        robot_draw = RobotRepresentation.RobotRepresentation(robot.getPosX(),
                                                             robot.getPosY(),
                                                             robot.getDirectionAngle(),
                                                             robot.width,
                                                             robot.length,
                                                             scaleFactor,
                                                             mode,
                                                             i)

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

        self.sonarShowing = True
        self.simShowing = True
        self.mode = args.mode


        self.robotRepresentations = initRobots(robots, args.scale_factor, args.mode)
        self.stations = stations#initStations(stations, args.scale_factor)

        self.painter = QPainter(self)

        self.initUI()

    def initUI(self):
        self.btSimulation = QPushButton(self)
        self.btSimulation.clicked.connect(self.clickedSimulation)
        self.btSimulation.move(0, 0)
        self.btSimulation.setFixedWidth(150)

        if self.mode == 'sonar':
            self.btSonar = QPushButton(self)
            self.btSonar.clicked.connect(self.clickedSonar)
            self.btSonar.move(160, 0)
            self.btSonar.setFixedWidth(120)

        self.updateButtons()

    def clickedSonar(self):
        if self.sonarShowing:
            self.sonarShowing = False
        elif not self.sonarShowing:
            self.sonarShowing = True
        self.updateButtons()

    def clickedSimulation(self):
        if self.simShowing:
            self.simShowing = False
        elif not self.simShowing:
            self.simShowing = True
        self.updateButtons()


    def updateButtons(self):

        if self.simShowing:
            self.btSimulation.setText("Visualisierung pausieren")
        elif not self.simShowing:
            self.btSimulation.setText("Visualisierung fortsetzen")

        #self.btSimulation.adjustSize()

        if self.mode == 'sonar':
            if self.sonarShowing:
                self.btSonar.setText("Sonar ausblenden")
            elif not self.sonarShowing:
                self.btSonar.setText("Sonar einblenden")
            #self.btSonar.adjustSize()



    def paintEvent(self, event):

        self.painter.begin(self)
        for station in self.stations:
            station.paint(self.painter)
        for robot in self.robotRepresentations:
            robot.paint(self.painter, self.sonarShowing)
        self.painter.end()

    def updateRobot(self, robot, num):
        self.robotRepresentations[num].update(robot.getPosX(), robot.getPosY(), robot.getDirectionAngle(), robot.radarHits, self.simShowing, robot.isActive(), robot.debugAngle)
        if self.simShowing:
            self.repaint()
        self.app.processEvents()
