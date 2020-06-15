from collections import namedtuple
import numpy as np
import keras as k
from keras.layers import Dense, Flatten, Input, Conv2D, ReLU

Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))

e1 = Experience(1,1,1,1)
e2 = Experience(2,2,2,2)
e3 = Experience(3,3,3,3)


experiences = [e1, e2, e3]

batch = Experience(*zip(*experiences))
print(batch)

states = np.extract(batch.state, batch)
print(states)

array = np.zeros(shape=(4, 3, 3))
print(array)

print("------------------\n------------------")
array2 = np.zeros(shape=(3, 3, 4))
print(array2)


input_shape = Input(shape=(3, 3, 4))
print(input_shape.shape)

