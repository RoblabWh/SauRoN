import Visualization.Components.RobotRepresentation as RobotRepresentation
from Environment.Components.Station import Station

from PyQt5.QtGui import QPainter, QFont
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QPushButton, QLabel, QSlider, QHBoxLayout
from PyQt5 import QtWidgets
import numpy as np
import time


def initRobots(robots, scaleFactor, mode, args):
    robotRepresentations = []
    for i, robot in enumerate(robots):
        robot_draw = RobotRepresentation.RobotRepresentation(robot.getPosX(),
                                                             robot.getPosY(),
                                                             robot.getDirectionAngle(),
                                                             robot.width,
                                                             robot.length,
                                                             scaleFactor,
                                                             mode,
                                                             i, args)

        robotRepresentations.append(robot_draw)
    return robotRepresentations


def initStations(stations, scaleFactor):
    _stations = []
    for i, station in enumerate(stations):
        station_draw = Station(station.posX, station.posY, station.width, station.length, i, scaleFactor)
        _stations.append(station_draw)
    return _stations


class SimulationWindow(QtWidgets.QMainWindow):

    def __init__(self, application, robots, stations, args, walls, circleWalls, arenaSize):
        super(SimulationWindow, self).__init__()

        self.args = args
        self.app = application
        self.setWindowTitle("Simulation")
        self.arenaWidth = arenaSize[0]
        self.arenaHeight = arenaSize[1]
        self.width = int(self.arenaWidth * args.scale_factor)
        self.height = int(self.arenaHeight * args.scale_factor)
        self.setGeometry(200, 100, self.width, self.height)

        self.sonarShowing = True
        self.simShowing = True
        self.SaveNetClicked = False
        self.mode = args.mode
        self.scaleFactor = args.scale_factor
        self.newScaleFactorWidth = self.geometry().width() / self.arenaWidth
        self.delay = 0
        self.selectedCategory = 0

        self.robotRepresentations = initRobots(robots, args.scale_factor, args.mode, self.args)
        self.stations = stations  # initStations(stations, args.scale_factor)
        self.walls = walls
        self.circleWalls = circleWalls

        #self.painter = QPainter(self)

        self.initUI()
        self.saveButtonListenrs = []
        self.monitorGraph = None

        self.app.aboutToQuit.connect(self.closeEvent)

    def resizeEvent(self, event):
        QtWidgets.QMainWindow.resizeEvent(self, event)
        self.resize()

    def resize(self):
        windowHeight = self.geometry().height()
        windowWidth = self.geometry().width()

        self.newScaleFactorWidth = windowWidth / self.arenaWidth

        self.setFixedHeight(self.arenaHeight * self.newScaleFactorWidth)

        self.scaleFactor = self.newScaleFactorWidth

        for robot in self.robotRepresentations:
            RobotRepresentation.RobotRepresentation.updateScale(robot, self.newScaleFactorWidth)

        for station in self.stations:
            Station.updateScale(station,
                                self.newScaleFactorWidth)

    def initUI(self):
        self.btSimulation = QPushButton(self)
        self.btSimulation.clicked.connect(self.clickedSimulation)
        self.btSimulation.setFixedWidth(150)

        self.lbSteps = QLabel(self)
        self.lbSteps.setText("0")
        self.lbSteps.setFont(QFont("Helvetica", 12, QFont.Black, ))
        self.lbSteps.setStyleSheet("color: rgba(0,0 ,0, 96);")

        self.btSaveNet = QPushButton(self)
        self.btSaveNet.clicked.connect(self.clickedSaveNet)
        self.btSaveNet.setFixedWidth(120)
        self.btSaveNet.setText("Netz speichern")

        hbox = QHBoxLayout()
        self.optionsWidget = QWidget(self)
        self.optionsWidget.setLayout(hbox)
        hbox.setSpacing(15)

        spacingWidget = QWidget(self)
        spacingWidget.setLayout(QHBoxLayout())

        if True:
            self.btSonar = QPushButton(self)
            self.btSonar.clicked.connect(self.clickedSonar)
            self.btSonar.setFixedWidth(120)

        if self.args.mode == 'test':
            self.slDelay = QSlider(Qt.Horizontal)
            self.slDelay.setRange(0, 100)
            self.slDelay.setValue(0)
            self.slDelay.setSingleStep(1)
            self.slDelay.setEnabled(True)
            self.slDelay.setFixedWidth(200)
            self.slDelay.valueChanged.connect(self.valueChangesSlider)

            lbSlTitle = QLabel(self)
            lbSlTitle.setText("Simulation Speed")
            lbSlTitle.setFont(QFont("Helvetica", 12, QFont.Black, ))
            lbSlTitle.setStyleSheet("color: rgba(0,0 ,0, 96);")
            lbSlLeft = QLabel(self)
            lbSlLeft.setText("normal")
            lbSlLeft.setFont(QFont("Helvetica", 9, QFont.Black, ))
            lbSlLeft.setStyleSheet("color: rgba(0,0 ,0, 96);")
            lbSlRight = QLabel(self)
            lbSlRight.setText("slow")
            lbSlRight.setFont(QFont("Helvetica", 9, QFont.Black, ))
            lbSlRight.setStyleSheet("color: rgba(0,0 ,0, 96);")

            sliderbox = QHBoxLayout()
            slWidget = QWidget(self)
            slWidget.setLayout(sliderbox)

            sliderbox.setSpacing(5)
            sliderbox.addWidget(lbSlTitle)
            sliderbox.addWidget(lbSlLeft)
            sliderbox.addWidget(self.slDelay)
            sliderbox.addWidget(lbSlRight)

            slWidget.setFixedWidth(500)

            hbox.addWidget(self.btSimulation)
            hbox.addWidget(self.btSonar)
            hbox.addWidget(self.btSaveNet)
            hbox.addWidget(slWidget)
            hbox.addWidget(spacingWidget)
            hbox.addWidget(self.lbSteps)

        else:
            hbox.addWidget(self.btSimulation)
            hbox.addWidget(self.btSonar)
            hbox.addWidget(self.btSaveNet)
            hbox.addWidget(spacingWidget)
            hbox.addWidget(self.lbSteps)

        self.setMenuWidget(self.optionsWidget)

        self.updateButtons()
        self.clickedSonar()

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


        if self.sonarShowing:
            self.btSonar.setText("Sonar ausblenden")
        elif not self.sonarShowing:
            self.btSonar.setText("Sonar einblenden")

    def paintEvent(self, event):

        painter = QPainter(self)

        # for station in self.stations:
        #     station.paint(painter)
        for i, robot in enumerate(self.robotRepresentations):
            self.stations[i].paint(painter)
            sonarShowing = self.sonarShowing
            if self.args.mode == 'test':
                if i != self.getActivationRobotIndex():
                    sonarShowing = False
            robot.paint(painter, sonarShowing)
        for wall in self.walls:
            wall.paint(painter, self.scaleFactor, self.args.display_normals)

        for circleWall in self.circleWalls:
            circleWall.paint(painter, self.scaleFactor)

        painter.end()

    def updateRobot(self, robot, num, stepsLeft, activations):
        if self.delay > 0: time.sleep(self.delay)

        self.robotRepresentations[num].update(robot.getPosX(), robot.getPosY(), robot.getDirectionAngle(), robot.lidarHits,
                                              self.simShowing, robot.isActive(), robot.debugAngle, activations,
                                              robot.getPieSliceWalls(), robot.posSensor)
        if self.simShowing:
            self.lbSteps.setText(str(stepsLeft))

    def paintUpdates(self):
        self.update()
        self.app.processEvents()

    def setWalls(self, walls):
        self.walls = walls

    def setCircleWalls(self, circleWalls):
        self.circleWalls = circleWalls

    def setSaveListener(self, observer):
        self.saveButtonListenrs.append(observer)

    def setRobotRepresentation(self, robots):
        self.robotRepresentations = initRobots(robots, self.scaleFactor, self.args.mode, self.args)
        for robot in self.robotRepresentations:
            RobotRepresentation.RobotRepresentation.updateScale(robot, self.newScaleFactorWidth)

    def setStations(self, stations):
        self.stations = stations
        for station in self.stations:
            Station.updateScale(station, self.newScaleFactorWidth)

    def getActivationRobotIndex(self):
        return 0

    def valueChangesSlider(self):
        self.delay = self.slDelay.value() / 500

    def setSize(self, arenaSize):
        self.arenaWidth = arenaSize[0]
        self.arenaHeight = arenaSize[1]
        self.width = int(self.arenaWidth * self.scaleFactor)
        self.height = int(self.arenaHeight * self.scaleFactor)
        self.newScaleFactorWidth = self.geometry().width() / self.arenaWidth
        self.setFixedHeight(self.arenaHeight * self.newScaleFactorWidth)

    def updateTrafficLights(self, proximity):
        self.selectedCategory = np.argmax(proximity)
