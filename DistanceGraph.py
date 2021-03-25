from collections import deque
import pyqtgraph as pg


class DistanceGraph:

    def __init__(self, application):
        self.app = application
        self.win = pg.GraphicsWindow()
        self.win.setWindowTitle("Distance Graph Roboter 0")

        self.datY = deque()
        self.datX = deque()

        self.p1 = self.win.addPlot(colspan=2)
        self.win.nextRow()

        self.curve1 = self.p1.plot()
        self.p1.setYRange(-0.1, 1.1)
        self.p1.setLabel(axis='left', text='Distance normalised')
        self.p1.setLabel(axis='bottom', text='sonar ray (half x-axis is forward)')

    def plot(self, valueToPlotX, valueToPlotY):
        self.datY = valueToPlotY
        self.datX = valueToPlotX
        self.curve1.setData(self.datX, self.datY)
        self.app.processEvents()