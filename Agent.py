import AIArenaSimulation

import random
import time


def updateValues():
    print("Begin" + str(AIArenaSimulation.soll_velocity))
    AIArenaSimulation.soll_velocity = random.randint(0, 10)
    print("Vel" + str(AIArenaSimulation.soll_velocity))
    AIArenaSimulation.angle = random.uniform(-180, 180)
    print("Angle" + str(AIArenaSimulation.angle))
    time.sleep(15)

updateValues()
