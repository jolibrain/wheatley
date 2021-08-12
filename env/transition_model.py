class TransitionModel:
    def __init__(self):
        self.graph = None

    def step(self, action):
        self.graph = None  # TODO change this

    def get_graph(self):
        return self.graph

    def done(self):
        return False

    def reset(self):
        pass
