class SetEval:
    def __init__(self, agent):
        self._agent = agent

    def __enter__(self):
        if hasattr(self._agent, 'set_eval'):
            self._agent.set_eval(True)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if hasattr(self._agent, 'set_eval'):
            self._agent.set_eval(False)

