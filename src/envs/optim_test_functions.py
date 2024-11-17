import numpy as np

from mushroom_rl.core import MDPInfo, Environment
from mushroom_rl.rl_utils import spaces


# https://en.wikipedia.org/wiki/Test_functions_for_optimization

class Quadratic(Environment):

    __name__ = 'Quadratic'

    def __init__(self, dim=2):
        self.dim = dim

        max_s = np.zeros(1)
        max_u = np.array([np.inf]*dim)

        # MDP properties
        observation_space = spaces.Box(low=-max_s, high=max_s)
        action_space = spaces.Box(low=-max_u, high=max_u)

        gamma = 1.
        horizon = 1
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        super().__init__(mdp_info)

    def __call__(self, u):
        return self.step(u)[1]

    def reset(self, state=None):
        return np.zeros(1)

    def step(self, u):
        u = np.atleast_2d(u)
        reward = np.linalg.norm(u, axis=1)**2
        reward *= -1
        return np.zeros(1), reward, True, {}

    def render(self):
        pass


class Beale(Environment):

    __name__ = 'Beale'

    def __init__(self):
        self.dim = 2

        max_s = np.zeros(1)
        max_u = np.array([np.inf] * self.dim)

        # MDP properties
        observation_space = spaces.Box(low=-max_s, high=max_s)
        action_space = spaces.Box(low=-max_u, high=max_u)

        gamma = 1.
        horizon = 1
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        super().__init__(mdp_info)

    def __call__(self, u):
        return self.step(u)[1]

    def reset(self, state=None):
        return np.zeros(1)

    def step(self, u):
        u = np.atleast_2d(u)
        a, b = u[:, 0], u[:, 1]
        reward = ((1.5 - a + a*b)**2 + (2.25 - a - a*(b**2))**2 + (2.625 - a - a*(b**3))**2)
        reward *= -1
        return np.zeros(1), reward, True, {}

    def render(self):
        pass


class Himmelblau(Environment):

    __name__ = 'Himmelblau'

    def __init__(self):
        self.dim = 2

        max_s = np.zeros(1)
        max_u = np.array([np.inf] * self.dim)

        # MDP properties
        observation_space = spaces.Box(low=-max_s, high=max_s)
        action_space = spaces.Box(low=-max_u, high=max_u)

        gamma = 1.
        horizon = 1
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        super().__init__(mdp_info)

    def __call__(self, u):
        return self.step(u)[1]

    def reset(self, state=None):
        return np.zeros(1)

    def step(self, u):
        u = np.atleast_2d(u)
        a, b = u[:, 0], u[:, 1]
        c = np.power(a, 2) + b - 11.
        d = a + np.power(b, 2) - 7
        reward = np.power(c, 2) + np.power(d, 2)
        reward *= -1
        return np.zeros(1), reward, True, {}

    def render(self):
        pass


class Rastrigin(Environment):

    __name__ = 'Rastrigin'

    def __init__(self, dim=2):
        self.dim = dim

        max_s = np.zeros(1)
        max_u = np.array([np.inf] * self.dim)

        # MDP properties
        observation_space = spaces.Box(low=-max_s, high=max_s)
        action_space = spaces.Box(low=-max_u, high=max_u)

        gamma = 1.
        horizon = 1
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        super().__init__(mdp_info)

    def __call__(self, u):
        return self.step(u)[1]

    def reset(self, state=None):
        return np.zeros(1)

    def step(self, u):
        u = np.atleast_2d(u)
        A = 10.
        fi = np.power(u, 2) - A * np.cos(2.0 * np.pi * u)
        reward = A * self.dim + np.sum(fi, axis=1)
        reward *= -1
        return np.zeros(1), reward, True, {}

    def render(self):
        pass


class Styblinski(Environment):

    __name__ = 'Styblinski'

    def __init__(self):
        self.dim = 2

        max_s = np.zeros(1)
        max_u = np.array([np.inf] * self.dim)

        # MDP properties
        observation_space = spaces.Box(low=-max_s, high=max_s)
        action_space = spaces.Box(low=-max_u, high=max_u)

        gamma = 1.
        horizon = 1
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        super().__init__(mdp_info)

    def __call__(self, u):
        return self.step(u)[1]

    def reset(self, state=None):
        return np.zeros(1)

    def step(self, u):
        u = np.atleast_2d(u)
        fi = np.power(u, 4.0) - 16.0 * np.power(u, 2) + 5 * u
        reward = 0.5 * np.sum(fi, axis=1)
        reward *= -1
        return np.zeros(1), reward, True, {}

    def render(self):
        pass


class Ackley(Environment):

    __name__ = 'Ackley'

    def __init__(self):
        self.dim = 2

        max_s = np.zeros(1)
        max_u = np.array([np.inf] * self.dim)

        # MDP properties
        observation_space = spaces.Box(low=-max_s, high=max_s)
        action_space = spaces.Box(low=-max_u, high=max_u)

        gamma = 1.
        horizon = 1
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        super().__init__(mdp_info)

    def __call__(self, u):
        return self.step(u)[1]

    def reset(self, state=None):
        return np.zeros(1)

    def step(self, u):
        u = np.atleast_2d(u)
        x, y = u[:, 0], u[:, 1]
        reward = -20. * np.exp(-0.2 * np.sqrt(0.5*(x**2 + y**2))) \
                 - np.exp(0.5*(np.cos(2*np.pi*x) + np.cos(2*np.pi*y))) + np.e + 20
        reward *= -1
        return np.zeros(1), reward, True, {}

    def render(self):
        pass


class Rosenbrock(Environment):

    __name__ = 'Rosenbrock'

    def __init__(self, dim=2):
        self.dim = dim

        max_s = np.zeros(1)
        max_u = np.array([np.inf] * self.dim)

        # MDP properties
        observation_space = spaces.Box(low=-max_s, high=max_s)
        action_space = spaces.Box(low=-max_u, high=max_u)

        gamma = 1.
        horizon = 1
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        super().__init__(mdp_info)

    def __call__(self, u):
        return self.step(u)[1]

    def reset(self, state=None):
        return np.zeros(1)

    def step(self, u):
        u = np.atleast_2d(u)
        fa = 100*(u[:, 1:] - u[:, :-1]**2)**2
        fb = (1 - u[:, :-1])**2
        reward = np.sum(fa + fb)
        reward *= -1
        return np.zeros(1), reward, True, {}

    def render(self):
        pass


class GoldsteinPrice(Environment):

    __name__ = 'GoldsteinPrice'

    def __init__(self):
        self.dim = 2

        max_s = np.zeros(1)
        max_u = np.array([np.inf] * self.dim)

        # MDP properties
        observation_space = spaces.Box(low=-max_s, high=max_s)
        action_space = spaces.Box(low=-max_u, high=max_u)

        gamma = 1.
        horizon = 1
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        super().__init__(mdp_info)

    def __call__(self, u):
        return self.step(u)[1]

    def reset(self, state=None):
        return np.zeros(1)

    def step(self, u):
        u = np.atleast_2d(u)
        x, y = u[:, 0], u[:, 1]
        reward = (1 + (x+y+1)**2*(19-14*x+3*x**2-14*y+6*x*y+3*y**2)) *\
                 (30+(2*x-3*y)**2*(18-32*x+12*x**2+48*y-36*x*y+27*y**2))
        reward *= -1
        return np.zeros(1), reward, True, {}

    def render(self):
        pass


class Booth(Environment):

    __name__ = 'Booth'

    def __init__(self):
        self.dim = 2

        max_s = np.zeros(1)
        max_u = np.array([np.inf] * self.dim)

        # MDP properties
        observation_space = spaces.Box(low=-max_s, high=max_s)
        action_space = spaces.Box(low=-max_u, high=max_u)

        gamma = 1.
        horizon = 1
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        super().__init__(mdp_info)

    def __call__(self, u):
        return self.step(u)[1]

    def reset(self, state=None):
        return np.zeros(1)

    def step(self, u):
        u = np.atleast_2d(u)
        x, y = u[:, 0], u[:, 1]
        reward = (x+2*y-7)**2 + (2*x+y-5)**2
        reward *= -1
        return np.zeros(1), reward, True, {}

    def render(self):
        pass


class Easom(Environment):

    __name__ = 'Easom'

    def __init__(self):
        self.dim = 2

        max_s = np.zeros(1)
        max_u = np.array([np.inf] * self.dim)

        # MDP properties
        observation_space = spaces.Box(low=-max_s, high=max_s)
        action_space = spaces.Box(low=-max_u, high=max_u)

        gamma = 1.
        horizon = 1
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        super().__init__(mdp_info)

    def __call__(self, u):
        return self.step(u)[1]

    def reset(self, state=None):
        return np.zeros(1)

    def step(self, u):
        u = np.atleast_2d(u)
        x, y = u[:, 0], u[:, 1]
        reward = -np.cos(x)*np.cos(y)*np.exp(-((x-np.pi)**2 + (y-np.pi)**2))
        reward *= -1
        return np.zeros(1), reward, True, {}

    def render(self):
        pass

