from models.agent import Agent
from env.env import Env


class Solver:
    def __init__(self, agent=None):
        if agent is None:
            env = Env(2, 2)
            self.agent = Agent(env)
        else:
            self.agent = agent

    @classmethod
    def load(cls, path):
        return cls(Agent.load(path), env=Env(2, 2))

    def train(self, training_schedule):
        n_epochs = training_schedule["n_epochs"]
        problem_trainings = training_schedule["problem_description"]
        for n in range(n_epochs):
            for problem_description, total_timesteps in problem_trainings:
                self.agent.train(problem_description, total_timesteps)

    def predict(self, problem_description):
        return self.agent(problem_description)
