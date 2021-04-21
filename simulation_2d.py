"""
Simulation of particles in a 2d box with gravity.

Author: Damian Hoedkte
Date: Apr 19, 2021

see https://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/

"""

import numpy as np
from numpy import asarray as arr
from scipy.spatial.distance import pdist, squareform

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

# constants
BOUNDS = [0, 0, 1, 1] # x0, y0, x1, y1
G = 9.81


class ParticleBox:
    """Particle box containing update function."""

    def __init__(self,
                 init_state=[[0., 0., 0., 0.]], # [xi, yi, vxi, vyi]
                 bounds=[0, 0, 0, 0], # x0, y0, x1, y1
                 ps=1.,
                 m=1.,
                 g=1.):
        self.ps = ps # particle size
        self.pr = self.ps * 0.5 # particle radius
        self.state = init_state.copy()
        self.bounds = bounds
        self.width = self.bounds[2] - self.bounds[0]
        self.height = self.bounds[3] - self.bounds[1]
        self.g = g
        self.time = 0.
        self.m = m * np.ones(self.state.shape[0])

    def step(self, dt):
        """calculate new state after dt"""
        self.time += dt

        # update positions
        self.state[:, :2] += self.state[:, 2:] * dt

        # find pairs of particles undergoing a collision
        D = squareform(pdist(self.state[:, :2]))
        ind1, ind2 = np.where(D < self.ps)
        unique = (ind1 < ind2)
        ind1 = ind1[unique]
        ind2 = ind2[unique]


        # update velocities of colliding pairs
        for k, l in zip(ind1, ind2):

        # iterate over particles
        #for k, (xk, yk, vxk, vyk) in enumerate(self.state[:-1]):

            #iterate over every other particles
        #    for l, (xl, yl, vxl, vyl) in enumerate(self.state[k+1:]):

            rk = self.state[k, :2] #arr([xk, yk])
            rl = self.state[l, :2] #arr([xl, yl])

            r_rel = rk - rl

            # calc distance
            s2 = np.dot(r_rel, r_rel)

            # check distance
            if (s2 > self.ps * self.ps):
                continue

            #-----------
            # collision
            #-----------

            # velocity vector
            vk = self.state[k, 2:] #arr([vxl, vyl])
            vl = self.state[l, 2:] #arr([vxk, vyk])

            #print(self.state[k, 2:], self.state[l, 2:])

            v_rel = vk - vl

            # in case collisions happend before

            d = self.ps - np.sqrt(s2)
            dtau = d / np.sqrt(np.dot(v_rel, v_rel))
            rk -= dtau * vk
            rl -= dtau * vl

            r_rel = rk - rl
            s2 = np.dot(r_rel, r_rel)

            # mass
            ml = self.m[l]
            mk = self.m[k]

            # momentum vector of the center of mass
            v_com = (ml * vl + mk * vk) * 1. / (ml + mk)

            # v in com
            vk_com = v_rel * mk * 1./(ml + mk)
            vl_com = -v_rel * ml * 1./(ml + mk)

            # collisions of spheres reflect v_rel over r_rel
            vk_com -= 2. * np.dot(vk_com, r_rel) * r_rel / s2
            vl_com -= 2. * np.dot(vl_com, r_rel) * r_rel / s2

            # assign new velocities
            self.state[k, 2:] = v_com + vk_com # * ml * 1. / (ml + mk)
            self.state[l, 2:] = v_com + vl_com # * mk * 1. / (ml + mk)

            #print(self.state[k, 2:], self.state[l, 2:])

            self.state[k, :2] = rk + (dt + dtau) * self.state[k, 2:]
            self.state[l, :2] = rl + (dt + dtau) * self.state[l, 2:]



        # check boundary conditions
        crossed_x0 = (self.state[:, 0] <= self.bounds[0] + self.pr)
        crossed_x1 = (self.state[:, 0] >= self.bounds[2] - self.pr)
        crossed_y0 = (self.state[:, 1] <= self.bounds[1] + self.pr)
        crossed_y1 = (self.state[:, 1] >= self.bounds[3] - self.pr)

        # set position to boundary
        self.state[crossed_x0, 0] = self.bounds[0] + self.pr
        self.state[crossed_x1, 0] = self.bounds[2] - self.pr
        self.state[crossed_y0, 1] = self.bounds[1] + self.pr
        self.state[crossed_y1, 1] = self.bounds[3] - self.pr

        # reflect velocity
        self.state[crossed_x0 | crossed_x1, 2] *= -1 # vx
        self.state[crossed_y0 | crossed_y1, 3] *= -1 # vy

        # add gravity
        self.state[:, 3] += - self.g * dt


#-------------------------------------------------------------------
# set up initial initial state
np.random.seed(0)
init_state = np.random.random((30, 4)) #arr([[0.4, 0.4, 0.01, 0.01], [0.6, 0.6, -0.01, -0.01]])
init_state[:, 2:] *= 0.1

scaled_G = G * 0.05 # corresponds to change of units m -> .. * m
box = ParticleBox(init_state=init_state, ps=0.03, m=1., bounds=BOUNDS, g=scaled_G)
dt = 1. / 70

#-------------------------------------------------------------------
# set up figure
# make boundaries a bit wider than box
ms_px = plt.rcParams['lines.markersize']
px = 1/plt.rcParams['figure.dpi']  # pixel in inches

fig = plt.figure(figsize=(600*px, 600*px)) # in inches


x0, y0 = arr(box.bounds[:2]) - box.width * 0.0 # square box
x1, y1 = arr(box.bounds[2:]) + box.width * 0.0

ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                    xlim=(x0, x1), ylim=(y0, y1))

plt.tick_params(axis='both', direction='in',
                    top=True, bottom=True, left=True, right=True,
                labelleft=False, labelbottom=False)

ms = 2*165 * box.ps

# plot particles
particles, = ax.plot([], [], 'bo', ms=ms)

rect = Rectangle(box.bounds[:2], box.width, box.height,
                 ec='k', fc='none', lw=2)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
ax.add_patch(rect)


# set up animation
def init():
    global box
    particles.set_data([], [])
    time_text.set_text('')
    return particles, time_text


def animate(i):
    """animation step"""
    global box, dt, ax
    box.step(dt)

    particles.set_data(box.state[:, 0], box.state[:, 1])
    time_text.set_text('%.2f' % box.time)

    return particles, time_text

animation = FuncAnimation(fig, animate, frames=800, interval=1000./70,
                            blit=True, init_func=init)

# show animation
plt.show()
