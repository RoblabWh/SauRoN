#!/bin/bash

ipadress=$(hostname -I | cut -d' ' -f1)
number=${ipadress: -3}

tensorboard --logdir=models/$number &
