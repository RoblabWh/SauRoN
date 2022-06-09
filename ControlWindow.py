from PyQt5 import QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

from algorithms.PPO_parallel.PPO_Multi import PPO_Multi


class ControlWindow(QtWidgets.QMainWindow):

    def __init__(self, application, nbOfEnvs, act_dim, env_dim, args, loadWeightsPath = ""):
        super(ControlWindow, self).__init__()
        #super().__init__()
        self.app = application
        self.args = args
        self.worker = None

        self.loadWeightsPath = loadWeightsPath

        self.setWindowTitle("Control Panel")

        self.model = PPO_Multi(act_dim, env_dim, args)
        self.currentEpisode = 0
        self.progressbarWidget = Progressbar(self.currentEpisode, self.args)
        self.tableWidget = Table(nbOfEnvs)


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

        self.startbutton = QPushButton("start")
        self.startbutton.clicked.connect(self.train)
        layout.addWidget(self.startbutton)

        self.setCentralWidget(self.widget)
        self.app.aboutToQuit.connect(self.closeEvent)

    def closeEvent(self, event):
        # if self.worker is not None:
        #     self.worker.quit()
        exit(0)

    def train(self):
        print("clicked")
        self.startbutton.setEnabled(False)
        self.tableWidget.fillTable()
        self.done, levelNames = self.model.prepare_training(self.loadWeightsPath)
        self.tableWidget.addLevelNames(levelNames)
        self.worker = WorkerThread(self.model, self.tableWidget.getVisibilites())
        self.worker.go = True
        self.worker.episode_done.connect(self.startNextSteps)
        self.worker.start()

        self.tableWidget.updateButtonsAtStart()

    def startNextSteps(self, episodeDone):
        self.tableWidget.updateButtons()
        #print("startNextSteps")
        if not self.done:
            closed_windows = self.model.get_closed_windows()
            self.tableWidget.updateButtonsOnWindowClosed(closed_windows)
            if not episodeDone:
                self.worker.update(self.model, self.tableWidget.getVisibilites())
                self.worker.go = True
            else:
                self.done, avrgRewardLastEpisode, successrates, currentEpisode, successAll = self.model.train_with_feedback_end_of_episode()
                self.currentEpisode = currentEpisode
                self.progressbarWidget.updateProgressbar(currentEpisode)
                self.tableWidget.updateAvrgRewardLastEpisode(avrgRewardLastEpisode)
                self.tableWidget.updateSuccessrate(successrates)
                self.successLabel.setText("Success insgesamt: " + str(successAll))


                if self.done is False:
                    self.worker.update(self.model, self.tableWidget.getVisibilites())
                    self.worker.go = True
                else:
                    print("Training done")
                    self.startbutton.setEnabled(True)
                    self.startbutton.setText("Ende")
                    self.startbutton.disconnect()
                    self.startbutton.clicked.connect(self.closeEvent)
                    print("Set start True")
                    if self.worker is not None:
                        self.worker.quit()
                        self.worker = None
        #print("end startNextSteps")

    def showandPause(self):
        self.show()
        self.app.exec_()

    def getTable(self):
        return self.tableWidget


class WorkerThread(QThread):
    episode_done = pyqtSignal(bool)

    def __init__(self, model, visibilities, parent = None):
        QThread.__init__(self, parent)
        super(WorkerThread, self)
        self.model = model
        self.visibilities = visibilities
        self.go = False

    def update(self, model, visibilities):
        self.visibilities = visibilities
        self.model = model

    def run(self):
        while True:
            if self.go is True:
                episodeDoneFuture = self.model.train_with_feedback_for_n_steps(self.visibilities)
                self.go = False
                self.episode_done.emit(episodeDoneFuture)


    

class Table(QWidget):
    def __init__(self, nbOfEnvs):
        super(Table, self).__init__()
        self.nbOfEnvs = nbOfEnvs
        self.headers = ["Env ID", "Level", "Reward", "Success", "Show/Hide Env"]
        self.buttonList = []
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

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.tabWidget)
        self.setLayout(self.layout)

        self.show()

    def fillTable(self):
        for env in range(self.nbOfEnvs):
            self.buttonList.append(QPushButton(self))
            self.tabWidget.setCellWidget(env, 4, self.buttonList[env])
            self.buttonList[env].clicked.connect(lambda *args, row=env, column=4: self.buttonClicked(row, column))

            self.tabWidget.setItem(env, 0, QTableWidgetItem(str(env)))
            self.tabWidget.setItem(env, 2, QTableWidgetItem("0"))
            self.tabWidget.setItem(env, 3, QTableWidgetItem("0"))


    def buttonClicked(self, row, col):

        # Change the text of the button
        if self.buttonList[row].text() == "Show Env":
            self.buttonList[row].setText("Initializing")
            self.levelVisibilty[row] = True
        elif self.buttonList[row].text() == "Hide":
            self.levelVisibilty[row] = False
            self.buttonList[row].setText("Hiding")

    def updateButtonsOnWindowClosed(self, list):
        for windowIndex in list:
            self.buttonList[windowIndex].setText("Show Env")
            self.levelVisibilty[windowIndex] = False

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

    def addLevelNames(self, levelNames):
        for i, name in enumerate(levelNames):
            self.tabWidget.setItem(i, 1, QTableWidgetItem(str(name)))

        for col in range(self.columns):
            self.tabWidget.resizeColumnToContents(col)


class Progressbar(QWidget):
    def __init__(self, currentEpisode, args):
        super(Progressbar, self).__init__()
        self.args = args
        self.progressbar = QProgressBar() # evt self übergeben

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


