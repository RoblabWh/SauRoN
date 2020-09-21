from collections import deque
import pyqtgraph as pg


class PlotterWindow:

    def __init__(self, application):
        self.app = application
        self.win = pg.GraphicsWindow()

        self.datY = deque()
        self.datX = deque()
        self.maxLen = 1000 # max number of data points to show on graph

        self.p1 = self.win.addPlot(colspan=2)
        self.win.nextRow()

        self.curve1 = self.p1.plot()
        self.p1.setYRange(-25, 25)
        self.p1.setLabel(axis='left', text='Geschwindigkeit in m/s')
        self.p1.setLabel(axis='bottom', text='Simulation Time in s')

    def plot(self, valueToPlotY, valueToPlotX):
        if len(self.datY) > self.maxLen:
            self.datY.popleft()  # remove oldest
        if len(self.datX) > self.maxLen:
            self.datX.popleft()
        self.datY.append(valueToPlotY)
        self.datX.append(valueToPlotX)
        self.curve1.setData(self.datX, self.datY)
        self.app.processEvents()