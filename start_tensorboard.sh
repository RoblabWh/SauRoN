#!/bin/bash

#ipadress=$(hostname -I | cut -d' ' -f1)
#number=${ipadress: -3}

tensorboard --logdir=models/101 --port 6006 &
tensorboard --logdir=models/102 --port 6007 &
tensorboard --logdir=models/103 --port 6008 &
tensorboard --logdir=models/104 --port 6009 &
