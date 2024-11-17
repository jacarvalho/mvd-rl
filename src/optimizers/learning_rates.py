import numpy as np
import numpy_ml as npml


class AdaptiveParameter(object):
    """
    This class implements a basic adaptive gradient step. Instead of moving of
    a step proportional to the gradient, takes a step limited by a given metric.
    To specify the metric, the natural gradient has to be provided. If natural
    gradient is not provided, the identity matrix is used.

    The step rule is:

    .. math::
        \\Delta\\theta=\\underset{\\Delta\\vartheta}{argmax}\\Delta\\vartheta^{t}\\nabla_{\\theta}J

        s.t.:\\Delta\\vartheta^{T}M\\Delta\\vartheta\\leq\\varepsilon

    Lecture notes, Neumann G.
    http://www.ias.informatik.tu-darmstadt.de/uploads/Geri/lecture-notes-constraint.pdf

    """
    def __init__(self, value):
        self._eps = value

    def __call__(self, *args, **kwargs):
        return self.update(*args, **kwargs)

    def update(self, *args, **kwargs):
        params = args[0]
        grads = args[1]
        lr = self.get_value(*args[1:], **kwargs)
        return params + lr * grads

    def get_value(self, *args, **kwargs):
        if len(args) == 2:
            gradient = args[0]
            nat_gradient = args[1]
            tmp = (gradient.dot(nat_gradient)).item()
            lambda_v = np.sqrt(tmp / (4. * self._eps))
            # For numerical stability
            lambda_v = max(lambda_v, 1e-8)
            step_length = 1. / (2. * lambda_v)

            return step_length
        elif len(args) == 1:
            return self.get_value(args[0], args[0], **kwargs)
        else:
            raise ValueError('Adaptive parameters needs gradient or gradient'
                             'and natural gradient')


class FixedLearningRate(object):
    """
    This class implements a fixed learning rate.

    """
    def __init__(self, value):
        self._lr = value

    def __call__(self, *args, **kwargs):
        params = args[0]
        grads = args[1]
        return self.update(params, grads)

    def update(self, params, grads):
        return params + self._lr * grads


class Adam(object):
    """
    This class implements the Adam learning rate scheme.

    """
    def __init__(self, value):
        self._optimizer = npml.neural_nets.optimizers.Adam(lr=value)

    def __call__(self, *args, **kwargs):
        params = args[0]
        grads = args[1]
        return self.update(params, grads)

    def update(self, params, grads):
        # -1*grads because numpy_ml does gradient descent, not ascent
        new_params = self._optimizer.update(params, -1.*grads, 'theta')
        return new_params

