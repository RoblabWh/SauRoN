import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

import numpy as np
import time
from pynput.keyboard import Listener


class BucketRenderer():
    """
    A BucketRenderer builds a continous bucket histogram. It takes an action, computes the bucket for a 10x10 bucket
    histogram and adds it to the current bucket. The bucket histogram is displayed in a matplotlib canvas.
    """
    def __init__(self, tail_num=15, delay=0.5):
        """
        Constructs a new BucketRenderer.
        :param tail_num: Number of actions tracked by the bucket renderer
        :param delay: Delay(in secs) between each rendering
        """
        plt.ion()
        plt.rcParams['toolbar'] = 'None'
        plt.rcParams['figure.facecolor'] = 'darkgrey'

        # algorithmic stuff
        self.tail_num = tail_num
        self.delay = delay
        self.bucket_hist = [[0 for _ in range(10)] for _ in range(10)]  # create bucket_histogram
        self.tail = []

        # figure stuff
        self.cb = None
        self.im = None
        self.fig = plt.figure(figsize=(5, 5), tight_layout=False)
        self.axes = self.fig.add_subplot(111)
        self.title = f'Last {self.tail_num} actions chosen by the NN.\nRed color means more activations\n Delay: {self.delay}'
        self.axes.set_title(self.title)
        self.axes.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.axes.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.axes.set_yticklabels([0.9, 0.7, 0.5, 0.3, 0.1, -0.1, -0.3, -0.5, -0.7, -0.9])
        self.axes.set_xticklabels([-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9])
        self.axes.set_ylabel('linVel')
        self.axes.set_xlabel('angVel')

        # keyboard interrupt stuff
        self.listener = Listener(on_press=self.set_delay)
        self.listener.start()

        plt.show(block=False)

    def show(self):
        """
        Displays the bucket histogram in a matplotlib canvas and sleeps for set delay.
        """
        if self.cb is not None:
            self.cb.remove()
        if self.im is not None:
            self.im.remove()
        self.im = self.axes.imshow(self.bucket_hist, cmap=cm.coolwarm, interpolation='none')
        #self.cb = self.fig.colorbar(self.im)
        self.fig.canvas.flush_events()
        #time.sleep(self.delay)

    def add_action(self, action):
        """
        Adds given action to the bucket histogram and pops tail if it's to long.
        :param action: 2D action between [-1, 1] in both dimensions
        """
        if len(self.tail) > self.tail_num:
            idx = self.tail.pop(0)
            self.bucket_hist[9 - idx[0]][9 - idx[1]] -= 1
        i, j = self.getBucket(action)
        self.bucket_hist[9 - i][9 - j] += 1
        self.tail.append((i, j))

    def getBucket(self, action):
        """
        Computes bucket for given action.
        :param action: 2D action between [-1, 1] in both dimensions
        """
        for i, irange in enumerate(range(-10, 10, 2)):
            irange_l = irange / 10  # build ranges
            irange_h = np.around(irange_l + 0.2, decimals=1)
            for j, jrange in enumerate(range(-10, 10, 2)):
                jrange_l = jrange / 10
                jrange_h = np.around(jrange_l + 0.2, decimals=1)
                if np.logical_and(np.logical_and(action[0] >= irange_l, action[0] <= irange_h),
                                  np.logical_and(action[1] >= jrange_l, action[1] <= jrange_h)):
                    return i, j

    def set_delay(self, key):
        """
        Callback function for pynput Listener
        :param key: pressed key
        :return:
        """

        if key.char == 'd':
            self.delay += 0.25
        if key.char == 'a':
            self.delay -= 0.25

        if self.delay < 0:
            self.delay = 0

        self.title = f'Last {self.tail_num} actions chosen by the NN.\nRed color means more activations\n Delay: {self.delay}'
        self.axes.set_title(self.title)