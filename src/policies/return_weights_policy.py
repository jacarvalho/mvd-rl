from mushroom_rl.policy.policy import ParametricPolicy


class ReturnWeightsPolicy(ParametricPolicy):
    """
    Policy that returns its weights when draw_action is called.
    Useful for testing functions in black-box optimization.

    """
    def __init__(self, mu):
        """
        Constructor.

        Args:
            mu (Regressor): the regressor representing the action to select
                in each state.

        """
        self._approximator = mu

        self._add_save_attr(_approximator='mushroom')

    def get_regressor(self):
        """
        Getter.

        Returns:
            the regressor that is used to map state to actions.

        """
        return self._approximator

    def __call__(self, state, action):
        raise NotImplementedError

    def draw_action(self, state):
        return self.get_weights()

    def set_weights(self, weights):
        self._approximator.set_weights(weights)

    def get_weights(self):
        return self._approximator.get_weights()

    @property
    def weights_size(self):
        return self._approximator.weights_size
