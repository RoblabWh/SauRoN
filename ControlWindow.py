from PyQt5 import QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import ray
import sys
from algorithms.DQN import DQN
#from algorithms.A2C_parallel.A2C_Multi import A2C_Multi
from algorithms.A2C_parallel.PPO_Multi import PPO_Multi
import EnvironmentWithUI

class ControlWindow(QtWidgets.QMainWindow):
    #TODO Fortschrittsbalken für alle Episoden
    # Tabelle mit 0 zB initialisieren
    # Tabelle übrige spaöten füllen
    # Werte runden
    # Spalten teilweise kleiner machen (falls das geht)


    def __init__(self, application, nbOfEnvs, act_dim, env_dim, args): #, model):  #TODO environments in einer Liste übergeben und speichern
        super(ControlWindow, self).__init__()
        ray.init()
        self.app = application


        self.setWindowTitle("Control Panel")
        self.setFixedSize(650, 700)

        self.model = PPO_Multi.remote(act_dim, env_dim, args)
        self.tableWidget = Table(nbOfEnvs)

        self.statusLabel = QLabel(self)
        self.statusLabel.setGeometry(0, 0, 550, 50)
        self.statusLabel.setWordWrap(True)
        self.statusLabel.move(0, 0)

        self.statusLabel.setFont(QFont("Helvetica", 15, QFont.Black))
        self.statusLabel.setText("Status Bar")  # TODO Get Data for status bar

        self.widget = QWidget(self)
        layout = QGridLayout()
        self.widget.setLayout(layout)
        layout.setSpacing(2)
        layout.setContentsMargins(0, 0, 0, 10)
        layout.addWidget(self.tableWidget)
        layout.addWidget(self.statusLabel)
        startbutton = QPushButton("start")
        startbutton.clicked.connect(self.train)
        layout.addWidget(startbutton)

        self.setCentralWidget(self.widget)


    def train(self):
        startup = self.model.prepareTraining.remote()
        self.done = ray.get(startup)
        self.worker = WorkerThread(self.model, self.tableWidget.getVisibilites())
        self.worker.start()
        self.worker.episode_done.connect(self.startNextSteps)
        self.tableWidget.updateButtonsAtStart()

    def startNextSteps(self, episodeDone):
        self.tableWidget.updateButtons()
        #TODO balken um eine episode erhöhen
        if not self.done:
            if not episodeDone:
                self.worker.terminate()
                self.worker = WorkerThread(self.model, self.tableWidget.getVisibilites())
                self.worker.start()
                self.worker.episode_done.connect(self.startNextSteps)
            else:
                doneFuture = self.model.trainWithFeedbackEpisodes.remote()
                self.done, avrgRewardLastEpisode = ray.get(doneFuture)
                self.tableWidget.updateAvrgRewardLastEpisode(avrgRewardLastEpisode)
                self.worker.terminate()
                self.worker = WorkerThread(self.model, self.tableWidget.getVisibilites())
                self.worker.start()
                self.worker.episode_done.connect(self.startNextSteps)



    def showandPause(self):
        self.show()
        self.app.exec_()

    def getTable(self):
        return self.tableWidget
    #
    # def getLevelVisibilities(self):
    #     return self.tableWidget.getVisibilites()

class WorkerThread(QThread):
    episode_done = pyqtSignal(bool)

    def __init__(self, model, visibilities, parent = None):
        QThread.__init__(self, parent)
        super(WorkerThread, self)
        self.model = model
        self.visibilities = visibilities

    def run(self):
        episodeDoneFuture = self.model.trainWithFeedbackSteps.remote(self.visibilities)
        episodeDone = ray.get(episodeDoneFuture)
        self.episode_done.emit(episodeDone)



class Table(QWidget):
    def __init__(self, nbOfEnvs):
        super(Table, self).__init__()
        self.nbOfEnvs = nbOfEnvs
        self.headers = ["Env ID", "Level", "Reward", "Success", "Show/Hide Env"]
        self.buttonList = []
        #self.model = model
        self.setTableLayout()
        self.levelVisibilty = [False for _ in range(nbOfEnvs)]
        self.levelVisibilty[0] = True


    def setTableLayout(self):
        self.tabWidget = QTableWidget()
        #self.tabWidget.setGeometry(0, 50, 400, 550)
        self.tabWidget.setMaximumWidth(650)
        self.tabWidget.setMaximumHeight(450)
        self.tabWidget.setRowCount(self.nbOfEnvs)
        self.tabWidget.setColumnCount(5)
        self.tabWidget.setHorizontalHeaderLabels(self.headers)
        self.tabWidget.verticalHeader().hide()

        for env in range(self.nbOfEnvs):
            self.buttonList.append(QPushButton(self))
            self.buttonList[env].setText("Waiting")
            self.buttonList[env].setEnabled(False)
            self.tabWidget.setCellWidget(env, 4, self.buttonList[env])
            self.buttonList[env].clicked.connect(lambda *args, row=env, column=4: self.buttonClicked(row, column))

            self.tabWidget.setItem(env, 0, QTableWidgetItem(str(env)))


        # for env in range(self.nbOfEnvs):
        #     self.btShowEnv = QPushButton(self)
        #     self.btShowEnv.setText("Show Env")
        #     self.tabWidget.setCellWidget(env, 4, self.btShowEnv)
        #     self.btShowEnv.clicked.connect(lambda *args, row=env, column=4: cellClick(row, column))

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.tabWidget)
        self.setLayout(self.layout)

        self.show()

    def buttonClicked(self, row, col):

        # Change the text of the button
        if self.buttonList[row].text() == "Show Env":
            self.buttonList[row].setText("Initializing")
            self.levelVisibilty[row] = True
        elif self.buttonList[row].text() == "Hide":
            self.levelVisibilty[row] = False
            self.buttonList[row].setText("Hiding")



        # TODO entsprechendes Fenster des Envs anzeigen (in separatem Fenster) in etwa wie: self.envsList[row].show()

    def getVisibilites(self):
        return self.levelVisibilty

    def updateButtons(self):
        for button in self.buttonList:
            if button.text() == "Initializing":
                button.setText("Hide")
            elif button.text() == "Hiding":
                button.setText("Show Env")

    def updateButtonsAtStart(self):
        for i, visible in enumerate(self.levelVisibilty):
            if visible:
                self.buttonList[i].setText("Hide")
            else:
                self.buttonList[i].setText("Show Env")
            self.buttonList[i].setEnabled(True)

    def updateAvrgRewardLastEpisode(self, avergRewards):
        for i, reward in enumerate(avergRewards):
            self.tabWidget.setItem(i, 2, QTableWidgetItem(str(reward)))






