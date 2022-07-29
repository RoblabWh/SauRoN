from pynput.keyboard import Listener
from algorithms.utils import AverageMeter
import numpy as np

class DebugListener():

    def __init__(self):
        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()
        self.averageMeter_tensor1 = AverageMeter()
        self.averageMeter_tensor2 = AverageMeter()
        self.step = 0
        self.step2 = 0
        self.listen = False
        self.listen2 = False

    def debug(self, var):
        if self.listen:
            self.step += 1
        if self.step >= 1:
            self.step = 0
            print(var)

    def debug2(self, var):
        self.averageMeter_tensor1.update(var.numpy()[0][0])
        self.averageMeter_tensor2.update(var.numpy()[0][1])
        if self.listen2:
            self.step2 += 1
        if self.step2 >= 25:
            self.step2 = 0
            print("Current sigma: ", var)
            print("Avg sigma: ", [self.averageMeter_tensor1.avg, self.averageMeter_tensor2.avg])
            print("Sum: ", np.sum(var.numpy()))

    def on_press(self, key):
        if key.char == 'l':
            self.listen = not self.listen
        if key.char == 'v':
            self.listen2 = not self.listen2

    def on_release(self, key):
        pass

