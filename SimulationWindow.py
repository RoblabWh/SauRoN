from PyQt5.QtGui import QPainter, QFont
from PyQt5.QtWidgets import QMainWindow, QWidget, QPushButton, QLabel
from PyQt5 import QtWidgets
import RobotRepresentation
from Station import Station
from Borders import ColliderLine
import DistanceGraph


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


class SimulationWindow(QtWidgets.QMainWindow):

    def __init__(self, application, robots, stations, args, walls):
        super().__init__()

        self.app = application
        self.setWindowTitle("Simulation")
        self.arenaWidth = args.arena_width
        self.arenaHeight = args.arena_length
        self.width = int(args.arena_width*args.scale_factor)
        self.height = int(args.arena_length*args.scale_factor)
        self.setGeometry(200, 100, self.width, self.height)
        # self.setFixedWidth(self.width)
        # self.setFixedHeight(self.height)
        #TODO scalefactor an fenstergröße binden
        self.sonarShowing = True
        self.simShowing = True
        self.SaveNetClicked = False
        self.mode = args.mode
        self.scaleFactor = args.scale_factor


        self.robotRepresentations = initRobots(robots, args.scale_factor, args.mode)
        self.stations = stations#initStations(stations, args.scale_factor)
        self.walls = walls

        self.painter = QPainter(self)

        self.initUI()
        self.saveButtonListenrs = []
        self.monitorGraph = None

        if (True):
            self.monitorGraph = DistanceGraph.DistanceGraph(application)


    def resizeEvent (self, event):
        windowHeight = self.geometry().height()
        windowWidth = self.geometry().width()

        QtWidgets.QMainWindow.resizeEvent(self, event)

        newScaleFactorWidth = windowWidth / self.arenaWidth

        self.setFixedHeight(self.arenaHeight * newScaleFactorWidth)

        self.scaleFactor = newScaleFactorWidth

        for robot in self.robotRepresentations:
            RobotRepresentation.RobotRepresentation.updateScale(robot, newScaleFactorWidth)

        for station in self.stations:
            Station.updateScale(station, newScaleFactorWidth)   # doof das wir alles anders importieren z.B. Station und RobotRepresentation -> TODO: alles einheitlich importieren



    def initUI(self):
        self.btSimulation = QPushButton(self)
        self.btSimulation.clicked.connect(self.clickedSimulation)
        self.btSimulation.move(0, 0)
        self.btSimulation.setFixedWidth(150)

        self.lbSteps = QLabel(self)
        self.lbSteps.setText("0")
        self.lbSteps.move(self.width - 68, 0)
        self.lbSteps.setFont(QFont("Helvetica", 14, QFont.Black, ))
        self.lbSteps.setStyleSheet("color: rgba(0,0 ,0, 96);")

        self.btSaveNet = QPushButton(self)
        self.btSaveNet.clicked.connect(self.clickedSaveNet)
        self.btSaveNet.move(290, 0)
        self.btSaveNet.setFixedWidth(120)
        self.btSaveNet.setText("Netz speichern")

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

    def clickedSaveNet(self):
        for observer in self.saveButtonListenrs:
            observer.saveCurrentWeights()


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
        for wall in self.walls:
            wall.paint(self.painter, self.scaleFactor)

        self.painter.end()

    def updateRobot(self, robot, num, stepsLeft):


        self.robotRepresentations[num].update(robot.getPosX(), robot.getPosY(), robot.getDirectionAngle(), robot.radarHits, self.simShowing, robot.isActive(), robot.debugAngle, robot.getPieSliceWalls(), robot.posSensor)
        if self.simShowing:
            observatedRobot = 0
            if self.monitorGraph != None and num == observatedRobot:
                distancesNormRob0 = robot.stateSonar[len(robot.stateSonar) - 1][0]
                self.monitorGraph.plot(range(len(distancesNormRob0)), distancesNormRob0)

            self.repaint()
            self.lbSteps.setText(str(stepsLeft))
        self.app.processEvents()



    def setWalls(self, walls):
        self.walls = walls

    def setSaveListener(self, observer):
        self.saveButtonListenrs.append(observer)



