from PyQt5.QtCore import QTimer
import Robot, PickUpStation, DeliveryStation
import PlotterWindow, SimulationWindow


class Simulation:

    def __init__(self, app):
        self.plotterWindow = PlotterWindow.PlotterWindow(app)
        self.simulationWindow = SimulationWindow.SimulationWindow()
        self.timer = QTimer()  # to create a thread that calls a function at intervals
        self.timer.timeout.connect(self.update)  # the update function keeps getting called at intervals
        self.timer.setSingleShot(False)

      #  self.simulationWindow = simulationWindow
        self.simulationWindow.show()
      #  self.plotterWindow = plotterWindow

        self.simTime = 0         # s
        self.simTimestep = 0.01  # s

        self.robot = Robot.Robot(5.0, 5.0, 0.0)
        self.pickUp = PickUpStation.PickUpStation(8.0, 1.0, 0.5, 0.5)
        self.delivery = DeliveryStation.DeliveryStation(1, 1, 0.5, 0.5)

        if self.simulationWindow != 0:
            self.simulationWindow.initRobot(meterToPixel(self.robot.getPosX()),
                                            meterToPixel(self.robot.getPosY()),
                                            self.robot.getDirection(),
                                            meterToPixel(self.robot.getWidth()),
                                            meterToPixel(self.robot.getLength()))

            self.simulationWindow.initPickUpStation(meterToPixel(self.pickUp.getPosX()),
                                                    meterToPixel(self.pickUp.getPosY()),
                                                    meterToPixel(self.pickUp.getWidth()),
                                                    meterToPixel(self.pickUp.getLength()))

            self.simulationWindow.initDeliveryStation(meterToPixel(self.delivery.getPosX()),
                                                      meterToPixel(self.delivery.getPosY()),
                                                      meterToPixel(self.delivery.getWidth()),
                                                      meterToPixel(self.delivery.getLength()))

        self.timer.start(self.simTimestep * 1000)



    def collideWithPickUp(self):
            # rechts, links, oben, unten
        if meterToPixel(self.robot.getPosX()) <= meterToPixel(self.pickUp.getPosX() + self.pickUp.getWidth()) and \
                meterToPixel(self.robot.getPosX() + self.robot.getWidth()) >= meterToPixel(self.pickUp.getPosX()) and \
                meterToPixel(self.robot.getPosY() + self.robot.getLength()) >= meterToPixel(self.pickUp.getPosY()) and \
                meterToPixel(self.robot.getPosY()) <= meterToPixel(self.pickUp.getPosY() + self.pickUp.getLength()):

            print("Found Pick Up Station")
            self.robot.setGoal(1)  # set Goal to be Delivery Station

    def collideWithDeliveryStation(self):

        if meterToPixel(self.robot.getPosX()) <= meterToPixel(self.delivery.getPosX() + self.delivery.getWidth()) and \
                meterToPixel(self.robot.getPosX() + self.robot.getWidth()) >= meterToPixel(self.delivery.getPosX()) and \
                meterToPixel(self.robot.getPosY() + self.robot.getLength()) >= meterToPixel(self.delivery.getPosY()) and \
                meterToPixel(self.robot.getPosY()) <= meterToPixel(self.delivery.getPosY() + self.delivery.getLength()):

            print("Found Delivery Station")

    def getRobot(self):
        return self.robot

    def getPickUpStation(self):
        return self.pickUp

    def getDeliveryStation(self):
        return self.delivery

    def getGoalX(self):
        if self.robot.getGoal() == 0:
            xGoal = self.pickUp.getPosX()
        elif self.robot.getGoal() == 1:
            xGoal = self.delivery.getPosX()
        return xGoal

    def getGoalY(self):
        if self.robot.getGoal() == 0:
            yGoal = self.pickUp.getPosY()
        if self.robot.getGoal() == 1:
            yGoal = self.delivery.getPosY()
        return yGoal

    def update(self):
        self.simTime += self.simTimestep
        print("Goal:" + str(self.robot.getGoal()))

        # nicht rechts oder links aus dem Fenster gehen
        if meterToPixel(self.robot.getPosX() + self.robot.width) > self.simulationWindow.width or meterToPixel(self.robot.getPosX()) < 0:
            print("out width, posX: " + str(self.robot.getPosX()))
            self.robot.setPose(5, 5)
            self.robot.linearVelocity = 0

        # nicht oben oder unten aus dem Fenster gehen
        if meterToPixel(self.robot.getPosY() + self.robot.length) > self.simulationWindow.height or meterToPixel(self.robot.getPosY()) < 0:
            print("out height, posY: " + str(self.robot.getPosY()))
            self.robot.setPose(5, 5)
            self.robot.linearVelocity = 0

        self.collideWithPickUp()
        self.collideWithDeliveryStation()
        self.robot.update(self.simTimestep)

        if self.simulationWindow != 0:
            self.simulationWindow.updateRobot(meterToPixel(self.robot.getPosX()),
                                              meterToPixel(self.robot.getPosY()),
                                              self.robot.getDirection())
        if self.plotterWindow != 0:
            self.plotterWindow.plot(self.robot.getLinearVelocity(), self.simTime)


def meterToPixel(m):
    return 100 * m
