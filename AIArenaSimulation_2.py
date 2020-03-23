import numpy as np
import pymunk
import pyglet
from pymunk.pyglet_util import DrawOptions
import pymunk.pyglet_util
import random
import time
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from multiprocessing import Process


options = DrawOptions()
WIDTH = 1280
HEIGHT = 720

numberOfPickUpStation = 1
numberOfDeliveryStation = 1
numberOfRobots = 1


window = pyglet.window.Window(WIDTH, HEIGHT, "AIArena Simulation", resizable=False)
space = pymunk.Space()
#space.gravity = 5, 10

# Robot
robotRadius = 20
robotBody = pymunk.Body(1, 1666)
robotBody.position = 640, 360
robotCircle = pymunk.Circle(robotBody, robotRadius)
robotCircle.color = (255, 255, 255)

# Fixed Robot Linear Velocity in x und y Richtung
#robotLinearVelocity = (random.randint(-50, 50), random.randint(-50, 50))
#robotBody.velocity = robotLinearVelocity   # alternative: robotBody._set_velocity(robotLinearVelocity)

# Fixed Robot Angular Velocity
#robotAngularVelocity = random.randint(-3, 3)
# robotBody.angular_velocity = robotAngularVelocity # alternative: robotBody._set_angular_velocity(robotAngularVelocity)

# Fixed Angle
# robotBody.angle = 10 # alternative: robotBody._set_angle(10)

# Random Impulse
# impulse = random.randint(-100, 100), random.randint(-100, 100)
# robotBody.apply_impulse_at_local_point(impulse)

# PickUp Station
pickUpStationSize = 40
pickUpStationBody = pymunk.Body(body_type=pymunk.Body.STATIC)
pickUpStationBody.position = 1000, 360
pickUpStationShape = pymunk.Poly.create_box(pickUpStationBody, (40, 40))
pickUpStationShape.id = 1
pickUpStationShape.color = (0, 200, 200)

# Delivery Station
deliveryStationBody = pymunk.Body(body_type=pymunk.Body.STATIC)
deliveryStationBody.position = 200, 360
deliveryStationShape = pymunk.Poly.create_box(deliveryStationBody, (40, 40))
deliveryStationShape.id = 2
deliveryStationShape.color = (0, 255, 0)

lineThickness = 2

# top line
segment1Body = pymunk.Body(body_type=pymunk.Body.STATIC)
segment1Shape = pymunk.Segment(segment1Body, (0, HEIGHT), (WIDTH, HEIGHT), lineThickness)
segment1Shape.id = 3
segment1Shape.color = (255, 0, 0)

# right line
segment2Body = pymunk.Body(body_type=pymunk.Body.STATIC)
segment2Shape = pymunk.Segment(segment1Body, (WIDTH, HEIGHT), (WIDTH, 0), lineThickness)
segment2Shape.id = 3
segment2Shape.color = (255, 0, 0)

# bottom line
segment3Body = pymunk.Body(body_type=pymunk.Body.STATIC)
segment3Shape = pymunk.Segment(segment1Body, (0, 0), (WIDTH, 0), lineThickness)
segment3Shape.id = 3
segment3Shape.color = (255, 0, 0)

# left line
segment4Body = pymunk.Body(body_type=pymunk.Body.STATIC)
segment4Shape = pymunk.Segment(segment1Body, (0, HEIGHT), (0, 0), lineThickness)
segment4Shape.id = 3
segment4Shape.color = (255, 0, 0)

testBody = pymunk.Body(1, 1666)
testBody.position = 400, 400
testCircle = pymunk.Circle(testBody, 20)
testCircle.color = (255, 255, 255)

space.add(robotBody, robotCircle, segment1Body, segment1Shape, segment2Body, segment2Shape, segment3Body, segment3Shape,
          segment4Body, segment4Shape, pickUpStationBody, pickUpStationShape, deliveryStationBody, deliveryStationShape,)


velocity = 0

# robot = [posX, posY, angle, linearVelocity, angularVelocity, acceleration, winkelAcceleration, targetX, targetY]


@window.event
def on_draw():
    window.clear()
    space.debug_draw(options)


# Collision Begin
def coll_begin(arbiter, space, data):
    global velocity
    print("Collision begin")

    # Found pick up station
    if arbiter.shapes[1].id == 1:
        print("Found Pick Up Station")
        updateReward(500)

    # Found delivery station
    elif arbiter.shapes[1].id == 2:
        print("Found Delivery Station")
        updateReward(500)

    # Collision with wall
    elif arbiter.shapes[1].id == 3:
        print("Crashed with wall")
        updateReward(-100)
        robotBody.position = 640, 360
        #velocity = 0
    else:
        pass

    return True


handler = space.add_default_collision_handler()
handler.begin = coll_begin


reward = 0

def updateReward(addReward):
    global reward
    reward += addReward
   # print("Reward: " + str(reward))
    return reward


def update(dt):
    # move(robotBody)

    steps = 0.04
    space.step(steps)

    move(robotBody)

    # Robot Linear Velocity in x und y Richtung -> muss 10 m/s sein
    #robotLinearVelocity = (random.randint(-50, 50), random.randint(-50, 50))
    robotLinearVelocity = (5, 10)
   # robotBody.velocity = robotLinearVelocity  # alternative: robotBody._set_velocity(robotLinearVelocity)
   # print(robotLinearVelocity)
    #time.sleep(5)

    # Robot Angular Velocity
    robotAngularVelocity = random.uniform(math.pi, -math.pi)
#    robotBody.angular_velocity = robotAngularVelocity
    #robotBody.angle = math.pi
    #print("angle: " + str(robotBody.angle))


def compute_next_position(position, velocity, timestep):
    return position + velocity * timestep


global soll_velocity


array_velocity = np.array([])
array_time_difference = np.array([])

#velocity = 8
i = 0
def compute_next_velocity(old_velocity):
    global array_velocity
    global array_time_difference

    global i
    i += 1
   # print("i" + str(i))
    global velocity
    max_linear_velocity = 10  # 10m/s
    max_acceleration = 2  # 2m/s^2
    max_brake_acceleration = -2
    dt = 0.01   # in sekunde
    global soll_velocity
    soll_velocity = 8  # soll_velocity wird eigentlich vom Netz übergeben

    # Test, mittendrin wird soll_velocity verringert --> bremsen
    if i >= 410:
        soll_velocity = 3


    # Beschleunigen, wenn old_velocity < soll_velocity
    if soll_velocity and old_velocity <= max_linear_velocity:
        if old_velocity < soll_velocity:
            velocity = old_velocity + max_acceleration * dt   # v(t) = v(t-1) + a * dt
        if velocity > max_linear_velocity:
            velocity = max_linear_velocity
        if velocity > soll_velocity:
            velocity = soll_velocity
    if soll_velocity > max_linear_velocity:
        soll_velocity = max_linear_velocity

    # Bremsen, wenn old_velocity > soll_velocity
    if soll_velocity and old_velocity <= max_linear_velocity:
        if old_velocity > soll_velocity:
            velocity = old_velocity + max_brake_acceleration * dt

   # time_difference = time.time() - start_time

    #array_time_difference = np.append(array_time_difference, time_difference)

    #array_velocity = np.append(array_velocity, velocity)
   # print("array_velocity" + str(array_velocity))


    print("velocity: " + str(velocity))
    print("old_velocity: " + str(old_velocity))
    print("soll_velocity: " + str(soll_velocity))
    return velocity


global angle
angle = 0
angle_radiant = 0


def compute_rotation(old_angle, soll_angle):
    global angle
    global angle_radiant
    dt = 0.01
    max_angle = math.pi
    min_angle = -math.pi
    angular_acceleration = 200


    # Test, mittendrin wird angle verändert
    if i >= 410:
        soll_angle = 320

    if i >= 650:
        soll_angle = 90

    # sinnvoll drehen, wenn Winkel > 180 dann rechts herum drehen
    if soll_angle > 180:
        soll_angle = soll_angle - 360

    if old_angle < soll_angle:
        angle = old_angle + angular_acceleration * dt
    if old_angle > soll_angle:
        angle = old_angle - angular_acceleration * dt


    angle_radiant = compute_radiant(angle)
    print("Actual Angle: " + str(compute_degrees(robotBody.angle)))
    print("Angle to be: " + str(soll_angle))
    return angle_radiant


def compute_degrees(angle_radiant):
    angle_deg = angle_radiant * 180 / math.pi
    return angle_deg


def compute_radiant(angle_degrees):
    angle_rad = angle_degrees * math.pi / 180
    return angle_rad



    
#    if soll_velocity <= max_linear_velocity:
#        if old_velocity > soll_velocity:
#            if old_velocity < max_linear_velocity:
#                velocity = old_velocity + max_brake_acceleration * dt
#            else:
#                velocity = max_linear_velocity
#    else:
#        soll_velocity = max_linear_velocity




'''
def compute_speed(velocity):
    max_linear_velocity = 10  # 10m/s
    max_acceleration = 2   # 2m/s^2
    dt = 0.01  # in sekunde
    # x_change = 0
    while velocity < max_linear_velocity:
        velocity = compute_next_velocity(velocity, max_acceleration, dt)
        #out_velocity = velocity / 10 * 5  # auf 0 bis 5 normalisiert
        print("out velocity: " + str(velocity))
    return velocity
'''


def move(body):



    # speed = 5
    # Koordinatentransformation
 #   body.position += (math.cos(body.angle) * compute_speed(velocity=0),
 #                     math.sin(body.angle) * compute_speed(velocity=0))
 #   print("position velocity: " + str(compute_speed(velocity=0)))

    global velocity

    body.position += (math.cos(body.angle) * compute_next_velocity(velocity),
                      math.sin(body.angle) * compute_next_velocity(velocity))

    compute_rotation(angle, 90)

    robotBody.angle = angle_radiant
   # print(body.angle)
   # print(body.position)


# old collision detection with wall
''' 
    # Crash with walls
    if (robotBody.position.x + robotRadius + lineThickness) >= WIDTH or (robotBody.position.x - robotRadius - lineThickness) <= 0:
        reward = -100
        robotBody.position = 640, 360
        print("Reward: " + str(reward) + ", Crash Wand: rechts oder links")
    elif (robotBody.position.y + robotRadius + lineThickness) >= HEIGHT or (robotBody.position.y - robotRadius - lineThickness) <= 0:
        reward = -100
        robotBody.position = 640, 360
        print("Reward: " + str(reward) + ", Crash Wand: oben oder unten")

'''

def plot_velocity():
    plt.plot(array_time_difference, array_velocity)
    plt.xlabel("Zeit t in Sekunden")
    plt.ylabel("Geschwindigkeit in m/s")
    plt.show()
    #if window.close():
    #    plt.close()


fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)


def animate(i):

    time_difference = time.time() - start_time
    ax1.clear()
    ax1.set_xlabel("Zeit t in Sekunden")
    ax1.set_ylabel("Geschwindigkeit in m/s")
    ax1.plot(time_difference, velocity)


def startSim():
    pyglet.clock.schedule_interval(update, 1.0 / 60)
    pyglet.app.run()



if __name__ == "__main__":
    global start_time
    start_time = time.time()
    ani = animation.FuncAnimation(fig, animate, interval=1000)
    plt.show()
    startSim()
    print("--- %s seconds --- " % (time.time() - start_time))
    #plot_velocity()




