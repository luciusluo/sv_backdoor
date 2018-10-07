# Particle Swarm Optimization writtten By Lucius

import random
import math


#--------------------- Class definition of a single particle ------------------------------------------#
class Particle:

    def __init__(self, boundary, dim):
        self.velocity = []
        self.position = []
        self.pBest = []


        for i in range(dim):
            self.velocity[i] = self.velocity.append(random.uniform(-1, 1))
            self.position[i] = self.position.append(random.uniform(boundary[i][0], boundary[i][1]))

        self.pBest = self.position


    def position_update(self, boundary):
        # Update position
        self.position = self.position + self.velocity

        # Check for exceeding boundary's max end
        for j in range(dim):
            if self.position[j] > boundary[j][1]:
                self.position[j] = bouddary[j][1]

        # Check for exceeding boundary's min end
        for k in range(dim):
            if self.position[k] < boundary[k][1]:
                self.position[k] = boundary[k][1]


    def velocity_update(self, gBest, v_Max):
        # Initialize parameters
        r1 = random.random()
        r2 = random.random()
        c1 = '%d'
        c2 = '%d'
        c3 = '%d'

        # Update velocity
        self.velocity = c1 * self.velocity + c2 * r1 * (self.pBest - self.position) + c3 * r2 * (gBest - self.pBest)

        # Check for exceeding max velocity
        for i in range(dim):
            if self.velocity[i] > v_Max:
                self.velocity[i] = v_Max


    def self_evaluate(self, func):
        if func(self.position) < func(self.pBest):
            self.pBest = self.position



#--------------------- Class definition of PSO of a swarm -------------------------------------------#
class PSO():

    global boun

    def __init__(self, boundary, dim, particle_num, terminate_err):

        swarm = []
        for i in range(dim):
            swarm[i] = Particle[parti]





    def eval_func(x):


#---------------------- Function that we try to optimize (minimize)----------------------------------#
def func(pos):
    return pso[0]^2 + pos[1]^2 + pos[2]


#---------------------- Run the program -------------------------------------------------------------#
dim = 3
particle_num = '%d'
terminate_err = "%d"      # When the largest bias among all the biases (position - gBest)
                          # of particles is smaller than this terminate_err, we conclude that
                          # all particles have converged to the gBest point.