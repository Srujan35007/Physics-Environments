import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import math
import pickle
plt.style.use('fivethirtyeight')
print(f'Imports complete')


def sin(x):
    return math.sin(x*np.pi/180)


def cos(x):
    return math.cos(x*np.pi/180)


class Rocket():
    def __init__(self, init_pos, init_vel):
        self.init_position = init_pos  # meters both pos_x and pos_y
        self.init_velocity = init_vel  # m/s both vel_x and vel_y
        self.gravity = 9.81  # N.m/s^2
        self.rocket_mass = 25000  # Kg

    def actions(self, n, prev_thrust, prev_gimbal):
        self.thrust0 = 0
        self.thrust100 = 845222  # N
        self.thrust50 = self.thrustMax*0.50  # N
        self.thrust30 = self.thrustMax*0.30  # N
        self.thrust20 = self.thrustMax*0.20  # N
        self.gimbal_left = -7.0  # Degrees engine turns right
        self.gimbal_right = 7.0  # Degrees engine turns left
        self.gimbal_zero = 0  # Degrees
        action_space = [self.thrust0, self.thrust100, self.thrust50, self.thrust30,
                        self.thrust20, self.gimbal_zero, self.gimbal_left, self.gimbal_right]
        thrust = prev_thrust
        gimbal = prev_gimbal
        if n <= 4:
            if action_space[n] == prev_thrust:
                thrust = prev_thrust
                gimbal = prev_gimbal
            else:
                thrust = action_space[n]
        else:
            if action_space[n] == prev_gimbal:
                gimbal = prev_gimbal
                thrust = prev_thrust
            else:
                gimbal = action_space[n]
        return (thrust, gimbal)

    def pos_vel(self, time_interval, thrust, gimbal, u_x, u_y, init_pos_x, init_pos_y):
        acc_x = thrust*sin(gimbal)/self.rocket_mass
        acc_y = ((thrust*cos(gimbal)) -
                 (self.rocket_mass*self.gravity))/self.rocket_mass
        vel_x = u_x + acc_x*time_interval
        vel_y = u_y + acc_y*time_interval
        pos_x = init_pos_x
        pos_y = init_pos_y
        pos_x += vel_x*time_interval
        pos_y += vel_y*time_interval
        observation = [(pos_x, pos_y), (vel_x, vel_y)]
        return observation

    def reward(self, observation, t):
        pos_x, pos_y = observation[0]
        vel_x, vel_y = observation[1]
        fire_time_reward = -t*10
        position_offset_reward = - \
            (math.sqrt(((pos_x-land_x)/1000)**2+((pos_y-land_y)/1000)**2))
        final_velocity_reward = - \
            (math.sqrt(((pos_y-land_y)/1000.0)**2)*(vel_y*100) +
             math.sqrt(((pos_x-land_x)/1000.0)**2)*(vel_x*100))
        if bool((pos_y - land_y)**2 < 100 and vel_y < 5 and vel_y > 0) and bool((pos_x - land_x)**2 < 10 and vel_x < 2 and vel_x > 0):
            done_reward = 1000
            done = True
            print('Done')
        else:
            done_reward = -1000
            done = False
        reward = fire_time_reward + position_offset_reward + \
            final_velocity_reward + done_reward
        return reward, done


time_interval = 0.25  # seconds
init_pos_x, init_pos_y = -3000, 41000
init_vel_x, init_vel_y = 10, -561
init_thrust = 0
init_gimbal = 0
land_x = 0
land_y = 0
N_ACTIONS = 8

rocket = Rocket((init_pos_x, init_pos_y), (init_vel_x, init_vel_y))

thrust = init_thrust
gimbal = init_gimbal
u_x = init_vel_x
u_y = init_vel_y
done = False

q_table = [] #---------------yet to be defined

while not done:
    for i in np.arange(0, 50+time_interval, time_interval):
        action = np.argmax(q_table[pos_x][pos_y][vel_x][vel_y])
        thrust, gimbal = rocket.actions(action, prev_thrust, prev_gimbal)
        observation = rocket.pos_vel(
            time_interval, thrust, gimbal, u_x, u_y, init_pos_x, init_pos_y)
        reward, done = rocket.reward(observation, i)
        # The actual training of the agent
        train() #------------yet to be defined
        pos_x, pos_y = observation[0]
        v_x, v_y = observation[1]
        u_x, u_y = v_x, v_y
        init_pos_x, init_pos_y = pos_x, pos_y

file_name = 'q_table.pickle'
pickle.dump(q_table, open(f'./{file_name}', 'wb'))
print('Q table saved.')

