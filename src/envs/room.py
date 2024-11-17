import math
import numpy as np
from copy import copy

from mushroom_rl.environments import Environment, MDPInfo
from mushroom_rl.utils import spaces

import pygame

from src.envs.corridor import Drawer


class Room(Environment):

    __name__ = 'Room'

    def __init__(self, gamma=0.99, horizon=1000, size=10, dt=1/30.,
                 pixel_size=200, change_shape=False,
                 change_color=False, image_mode=False):

        self._image_mode = image_mode

        max_obs = np.array([size, size])
        observation_space = spaces.Box(low=np.zeros_like(max_obs), high=max_obs)
        self._max_action = np.array([1., 1.])
        action_space = spaces.Box(low=-self._max_action,
                                  high=self._max_action)
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        self._size = size
        self._dt = dt
        self._vel = 25.
        self._agent_size = size / 10
        self._goal_area_radius = 0.5 * 5.0 * 10. * self._dt

        self._b1_lower_right = [3., 2.]
        self._b2_upper_left = [6.0, 8.0]


        self._agent_position = None
        self._agent_shape = None if change_shape else 0

        self._goal_position = np.array([1.00 * size, 1.00 * size])

        self._colors = [
            (0, 0, 0),
            (255, 255, 255),
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (210, 210, 80),
            (153, 76, 0)
        ]

        self._color_idx = None if change_color else np.arange(4)

        self._viewer = Drawer(size, size,
                              width=pixel_size,
                              height=pixel_size)

        self._context_list = list()
        self._forced_context = None

        self._fixed_reset_agent_position = True

        super().__init__(mdp_info)

    def reset(self, state=None):

        self._agent_shape = 0
        self._color_idx = np.array([0, 1, 2, 3])

        self._sample_agent_position()

        if self._image_mode:
            return self._draw()
        return self._agent_position

    def step(self, action):
        if isinstance(action, int):
            if action == 0:
                action = np.array([0., 0.2])
            elif action == 1:
                action = np.array([0., -0.2])
            elif action == 2:
                action = np.array([-0.2, 0.])
            elif action == 3:
                action = np.array([0.2, 0.])

        action = self._bound(action, -self._max_action, self._max_action)
        u = action
        act = u * self._vel

        new_pos = copy(self._agent_position + act * self._dt)

        collided_external_walls = self._collision_external_walls(new_pos)
        collided_box = self._collision_box(new_pos)
        if not collided_external_walls:
            self._agent_position = new_pos

        img = self._draw()

        dist = np.linalg.norm(self._goal_position - self._agent_position)
        reward = -dist ** 2 / 100

        absorbing = dist < self._goal_area_radius
        if absorbing:
            reward += 10
            # reward += 0

        if collided_box or collided_external_walls:
            reward += -100
            # absorbing = False

        return self._agent_position, reward, absorbing, {}

    def render(self):
        self._viewer.display(self._dt)

    def stop(self):
        self._viewer.close()

    def _sample_agent_position(self):
        agent_position = np.zeros(2)

        agent_position[0] = 0.5
        agent_position[1] = 1.

        # agent_position[0] = 0.5
        # agent_position[1] = self._size/2

        self._agent_position = agent_position

    def _collision_external_walls(self, new_pos):
        xp = new_pos[0]
        yp = new_pos[1]
        if xp <= 0. or xp >= self._size:
            return True
        if yp <= 0. or yp >= self._size:
            return True

        return False

    def _collision_box(self, new_pos):
        xp = new_pos[0]
        yp = new_pos[1]
        # collision box 1
        if (0. <= xp <= self._b1_lower_right[0] + 0.5) and (self._b1_lower_right[1] <= yp <= self._size):
            return True
        # collision box 2
        if (self._b2_upper_left[0] <= xp <= self._size) and (0 <= yp <= self._b2_upper_left[1]):
            return True

        return False

    def _draw(self):
        background_color = self._colors[self._color_idx[0]]
        wall_color = self._colors[self._color_idx[1]]
        goal_color = self._colors[self._color_idx[2]]
        agent_color = self._colors[self._color_idx[3]]

        self._viewer.screen.fill(background_color)

        self._viewer.circle(self._goal_position, self._goal_area_radius, goal_color)

        # block 1
        w = self._b1_lower_right[0]
        h = self._size - self._b1_lower_right[1]
        a = math.ceil(w/2)
        b = math.ceil(h/2)
        center = np.array([(0 + self._b1_lower_right[0])/2,
                           self._b1_lower_right[1] + (self._size - self._b1_lower_right[1])/2])
        points = [[-a, b],
                  [a, b],
                  [a, -b],
                  [-a, -b]
                  ]

        self._viewer.polygon(center, 0, points, color=wall_color)

        # block 2
        w = self._size - self._b2_upper_left[0]
        h = self._b2_upper_left[1]
        a = math.ceil(w/2)
        b = math.ceil(h/2)
        center = np.array([self._b2_upper_left[0] + (self._size - self._b2_upper_left[0]) / 2,
                           self._b2_upper_left[1] / 2])
        points = [[-a, b],
                  [a, b],
                  [a, -b],
                  [-a, -b]
                  ]

        self._viewer.polygon(center, 0, points, color=wall_color)

        self._viewer.circle(self._agent_position, self._agent_size/5, color=agent_color)

        img = pygame.surfarray.array3d(self._viewer.screen)
        img = img.swapaxes(0, 1)
        img = np.rollaxis(img, 2, 0)

        return img


if __name__ == '__main__':
    from pygame.locals import *


    def get_key(easy):
        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    exit()
                    easy = True

            pressed = pygame.key.get_pressed()
            keys = [K_UP, K_DOWN, K_LEFT, K_RIGHT] if easy else range(K_KP1, K_KP9 + 1)
            for i, key in enumerate(keys):
                if pressed[key]:
                    return i


    easy = True
    image_mode = True
    easy_task = True
    mdp = Room(image_mode=image_mode)


    while True:
        mdp.reset()
        if image_mode:
            mdp.render()

        absorbing = False
        returns = 0
        steps = 0
        gamma = 0.99
        i = 0
        while not absorbing:
            a = get_key(easy)
            # a = np.random.randn(2) * 1.0
            state, reward, absorbing, _ = mdp.step(a)
            print(state, reward)
            returns += reward * gamma**i
            steps += 1
            mdp.render()
            if absorbing:
                print(state, returns, steps)

            i += 1
