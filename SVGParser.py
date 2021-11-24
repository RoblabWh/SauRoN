import math
import xml.etree.ElementTree as ET
import random

import Borders
import Robot
import Station


class SVGLevelParser:

    def __init__(self, filename, args):


        file = 'svg/'+filename
        self.stations, self.robots, self.lines, self.circles = [], [], [], []

        dpiFactor = 1/28.35 #72 dotsPerInch converted  into DotsPerCentimeter
        #dpiFactor /= 10
        # svg = SVG.parse(file)
        # list(svg.elements())
        # print(svg.elements())

        svg = ET.parse(file)
        svg = svg.getroot() # kann man weglassen
        # print(svg.attrib)
        # print(svg.attrib['width'])
        # # svg = ET.fromstring("svg/hall.svg")
        # print(svg)
        rects = svg.findall('.//{http://www.w3.org/2000/svg}rect')
        paths = svg.findall('.//{http://www.w3.org/2000/svg}path')
        polylines = svg.findall('.//{http://www.w3.org/2000/svg}polyline')
        polygons = svg.findall('.//{http://www.w3.org/2000/svg}polygon')
        circles = svg.findall('.//{http://www.w3.org/2000/svg}circle')
        lines = svg.findall('.//{http://www.w3.org/2000/svg}line')
        # print(rects)
        # print(paths)
        # print(circles)

        tareq = 'tareq' in filename

        self.arenaSize = svg.attrib['viewBox'].split(' ')[2:]
        self.arenaSize = [float(self.arenaSize[0])*dpiFactor, float(self.arenaSize[1])*dpiFactor]


        for rect in rects:
            attributes = rect.attrib
            show = True
            if 'display' in attributes:
                if rect.attrib['display'] == 'none':
                    show = False

            if show:
                width = float(rect.attrib['width'])*dpiFactor
                height = float(rect.attrib['height'])*dpiFactor
                x = float(rect.attrib['x'])*dpiFactor + width*.5
                y = float(rect.attrib['y'])*dpiFactor + height*.5
                # print(rect, width, height, x, y)
                if 'transform' in rect.attrib:
                    rectWall = Borders.SquareWall(x, y, width, height)
                    transform = rect.attrib['transform'].split('(')[1:]
                    transform = transform[0].split()
                    rectWall.rotate(float(transform[0]), float(transform[1]), float(transform[2]), float(transform[3]))
                    # rectWall.rotate(0, -1, 1, 0)
                    # print(transform)
                    self.lines+= rectWall.getBorders()

                else:
                    self.lines += Borders.SquareWall(x, y, width, height).getBorders()

        for polyline in polylines:
            attributes = polyline.attrib
            show = True
            if 'display' in attributes:
                if polyline.attrib['display'] == 'none':
                    show = False

            if show:

                points = polyline.attrib['points'].split()
                pointsLen = len(points)
                points = [points[i].split(',') for i in range(0, pointsLen)]

                for i in range (0, pointsLen-1):
                    x1 = float(points[i][0]) * dpiFactor
                    y1 = float(points[i][1]) * dpiFactor
                    x2 = float(points[(i+1)][0]) * dpiFactor
                    y2 = float(points[(i+1)][1]) * dpiFactor
                    # print(polyline, x1, y1, x2, y2)
                    self.lines += [Borders.ColliderLine(x1, y1, x2, y2)]

        for polygon in polygons:
            attributes = polygon.attrib
            show = True
            if 'display' in attributes:
                if polygon.attrib['display'] == 'none':
                    show = False

            if show:

                points = polygon.attrib['points'].split()
                pointsLen = len(points)
                points = [points[i].split(',') for i in range(0, pointsLen)]

                for i in reversed(range(0, pointsLen)):
                    x1 = float(points[i][0]) * dpiFactor
                    y1 = float(points[i][1]) * dpiFactor
                    x2 = float(points[(i-1) % pointsLen][0]) * dpiFactor
                    y2 = float(points[(i-1) % pointsLen][1]) * dpiFactor
                    # print(polygon, x1, y1, x2, y2)
                    self.lines += [Borders.ColliderLine(x1, y1, x2, y2)]


        for line in lines:
            attributes = line.attrib
            show = True
            if 'display' in attributes:
                if line.attrib['display'] == 'none':
                    show = False

            if show:
                #id = line.attrib['id']
                x1 = float(line.attrib['x1']) * dpiFactor
                y1 = float(line.attrib['y1']) * dpiFactor
                x2 = float(line.attrib['x2']) * dpiFactor
                y2 = float(line.attrib['y2']) * dpiFactor
                # print(line, id, x1, y1, x2, y2)
                if tareq:
                    self.lines += [Borders.ColliderLine(x2, y2, x1, y1)]
                else:
                    self.lines += [Borders.ColliderLine(x1, y1, x2, y2)]

        stationsData = []
        self.robotsData = []
        startAndGoalCircles = [] # only used if circles are not defined as start and goal by an ID
        for circle in circles:
            attributes = circle.attrib
            show = True
            if 'display' in attributes:
                if circle.attrib['display'] == 'none':
                    show = False

            if show:
                cx = float(circle.attrib['cx']) * dpiFactor
                cy = float(circle.attrib['cy']) * dpiFactor
                r = float(circle.attrib['r']) * dpiFactor
                if 'id' in attributes:
                    id= circle.attrib['id']

                    # print(circle, cx, cy, r, id)
                    if 'start' in id:
                        self.robotsData += [(cx, cy)]

                    elif 'goal' in circle.attrib['id']:
                        stationsData += [(cx, cy, r)]
                elif 'stroke' in attributes:
                    strokes = circle.attrib['stroke']
                    if circle.attrib['stroke'] == 'orange':
                        startAndGoalCircles += [(cx, cy, r)]
                    else:
                        self.circles += [Borders.CircleWall(cx, cy, r)]
                else:
                    self.circles += [Borders.CircleWall(cx, cy, r)]

        #create goals and starts for 4 roboters by randomly sampling a position defined by the level svg file
        for i in range(4):
            if len(startAndGoalCircles)>2:
                startIndex = random.randint(0, len(startAndGoalCircles)-1)
                startR = startAndGoalCircles.pop(startIndex)
                self.robotsData += [(startR[0], startR[1])]

                goalIndex = random.randint(0, len(startAndGoalCircles)-1)
                goal = startAndGoalCircles.pop(goalIndex)
                stationsData += [goal]


        for i, data in enumerate(stationsData):
            self.stations += [Station.Station(data[0], data[1], data[2], i, args.scale_factor)]
        # random.shuffle(self.stations)

        self.stationsData = [stationsData[i][:-1] for i in range(0,len(stationsData))]

        for i, data in enumerate(self.robotsData):
            self.robots += [Robot.Robot((data[0], data[1]), 0, self.stations[i], args, self.lines, self.stations, self.circles)]

    def getRobots(self):
        return self.robots

    def getStations(self):
        return self.stations

    def getWalls(self):
        return self.lines

    def getCircleWalls(self):
        return self.circles

    def getRobsPos(self):
        return self.robotsData

    def getRobsOrient(self):
        return [random.random()*2*math.pi for _ in range(len(self.robots))]

    def getStatsPos(self):
        return self.stationsData

    def getArenaSize(self):
        return self.arenaSize


def getBorders(self):
        return (self.lines, self.circles)