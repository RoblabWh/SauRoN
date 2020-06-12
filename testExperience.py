from collections import namedtuple
import numpy as np


Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))

e1 = Experience(1,1,1,1)
e2 = Experience(2,2,2,2)
e3 = Experience(3,3,3,3)


experiences = [e1, e2, e3]

batch = Experience(*zip(*experiences))
print(batch)

states = np.extract(batch.state, batch)
print(states)


