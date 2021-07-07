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
    #TODO Fortschrittsbalken für alle Episoden -> erledigt
    # Tabelle mit 0 zB initialisieren -> erledigt
    # Tabelle übrige spaöten füllen -> fehlt nur noch Level -> was soll da genau drin stehen: Ein Name?
    # Werte runden -> erledigt
    # Spalten teilweise kleiner machen (falls das geht)


    def __init__(self, application, nbOfEnvs, act_dim, env_dim, args, loadWeightsPath = ""): #, model):  #TODO environments in einer Liste übergeben und speichern
        super(ControlWindow, self).__init__()
        ray.init()
        self.app = application
        self.args = args

        self.loadWeightsPath = loadWeightsPath

        self.setWindowTitle("Control Panel")
        self.setFixedSize(350, 700)

        self.model = PPO_Multi.remote(act_dim, env_dim, args)
        self.currentEpisode = 0
        self.progressbarWidget = Progressbar(self.currentEpisode, self.args)
        self.tableWidget = Table(nbOfEnvs)


        #self.progressbarWidget.setGeometry()

        # self.statusLabel = QLabel(self)
        # self.statusLabel.setGeometry(0, 0, 550, 50)
        # self.statusLabel.setWordWrap(True)
        # self.statusLabel.move(0, 0)

        # self.statusLabel.setFont(QFont("Helvetica", 15, QFont.Black))
        # self.statusLabel.setText("Status Bar")  # TODO Get Data for status bar

        self.successLabel = QLabel(self)
        self.successLabel.setFont(QFont("Helvetica", 12, QFont.Black))
        self.successLabel.setText("Success insgesamt: ")

        self.widget = QWidget(self)
        layout = QGridLayout()
        self.widget.setLayout(layout)
        layout.setSpacing(2)
        layout.setContentsMargins(0, 0, 0, 10)

        layout.addWidget(self.tableWidget)
        layout.addWidget(self.successLabel)
        layout.addWidget(self.progressbarWidget)

        #layout.addWidget(self.statusLabel)

        startbutton = QPushButton("start")
        startbutton.clicked.connect(self.train)
        layout.addWidget(startbutton)

        self.setCentralWidget(self.widget)


    def train(self):
        startup = self.model.prepareTraining.remote(self.loadWeightsPath)
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
                self.done, avrgRewardLastEpisode, successrates, currentEpisode, successAll = ray.get(doneFuture)
                self.currentEpisode = currentEpisode
                self.progressbarWidget.updateProgressbar(currentEpisode)
                self.tableWidget.updateAvrgRewardLastEpisode(avrgRewardLastEpisode)
                self.tableWidget.updateSuccessrate(successrates)
                self.successLabel.setText("Success insgesamt: " + str(successAll))
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
        self.tabWidget.setMaximumWidth(450)
        self.tabWidget.setMaximumHeight(450)
        self.tabWidget.setRowCount(self.nbOfEnvs)
        self.tabWidget.setColumnCount(5)
        self.tabWidget.setHorizontalHeaderLabels(self.headers)
        self.tabWidget.verticalHeader().hide()
        self.tabWidget.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)

        self.columns = 4
        for col in range(self.columns):
            self.tabWidget.resizeColumnToContents(col)


        for env in range(self.nbOfEnvs):
            self.buttonList.append(QPushButton(self))
            self.buttonList[env].setText("Waiting")
            self.buttonList[env].setEnabled(False)
            self.tabWidget.setCellWidget(env, 4, self.buttonList[env])
            self.buttonList[env].clicked.connect(lambda *args, row=env, column=4: self.buttonClicked(row, column))

            self.tabWidget.setItem(env, 0, QTableWidgetItem(str(env)))
            self.tabWidget.setItem(env, 2, QTableWidgetItem("0"))
            self.tabWidget.setItem(env, 3, QTableWidgetItem("0"))




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
            rewardRounded = round(reward, 5)
            self.tabWidget.setItem(i, 2, QTableWidgetItem(str(rewardRounded)))

        for col in range(self.columns):
            self.tabWidget.resizeColumnToContents(col)

    def updateSuccessrate(self, successrates):
        for i, successrate in enumerate(successrates):
            self.tabWidget.setItem(i, 3, QTableWidgetItem(str(successrate)))

        for col in range(self.columns):
            self.tabWidget.resizeColumnToContents(col)


class Progressbar(QWidget):
    def __init__(self, currentEpisode, args):
        super(Progressbar, self).__init__()
        self.args = args
        self.progressbar = QProgressBar() # evt self übergeben
        #self.progressbar.setValue(50)
        #self.progressbar.setFixedWidth(450)
        #self.progressbar.setGeometry(0,0,300,25)
        #self.progressbar.setFixedSize(650, 200)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.progressbar)
        self.setLayout(self.layout)

        self.updateProgressbar(currentEpisode)

        self.show()

    def updateProgressbar(self, currentEpisode):

        value = (currentEpisode / self.args.nb_episodes) * 100
        self.progressbar.setValue(value)
        self.progressbar.setFormat(str(currentEpisode) + " / " + str(self.args.nb_episodes) + " Episoden")
        self.progressbar.setAlignment(Qt.AlignCenter)


