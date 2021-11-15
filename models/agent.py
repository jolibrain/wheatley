import pickle

from stable_baselines3.common.callbacks import EveryNTimesteps
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import PPO
from stable_baselines3.a2c import A2C
import torch

from env.env import Env
from models.agent_callback import ValidationCallback
from models.policy import Policy
from models.features_extractor import FeaturesExtractor
from problem.problem_description import ProblemDescription


class Agent:
    def __init__(
        self,
        env_specification,
        model=None,
        agent_specification=None,
    ):
        """
        There are 2 ways to init an Agent:
         - Either provide a valid env_specification and agent_specification
         - Or use the load method, to load an already saved Agent
        """

        # User must provide an agent_specification or a model at least.
        if agent_specification is None and model is None:
            raise Exception("Please provide an agent_specification or a model to create a new Agent")

        # If a model is provided, we simply load the existing model.
        if model is not None:
            self.model = model
            self.env_specification = env_specification
            return

        # Else, we have to build a new PPO instance.
        # We have to create a fake_env to instantiate PPO.
        fake_problem_description = ProblemDescription(
            transition_model_config="L2D",
            reward_model_config="Sparse",
            fixed=True,
            deterministic=True,
            n_jobs=2,
            n_machines=2,
            max_duration=99,
        )
        env_fns = [lambda: Env(fake_problem_description, env_specification) for _ in range(agent_specification.n_workers)]
        fake_env = DummyVecEnv(env_fns)

        # Finally, we can build our PPO
        self.model = PPO(
            Policy,
            fake_env,
            learning_rate=agent_specification.lr,
            n_steps=agent_specification.n_steps_episode,
            batch_size=agent_specification.batch_size,
            n_epochs=agent_specification.n_epochs,
            gamma=agent_specification.gamma,
            gae_lambda=1,  # To use same vanilla advantage function
            clip_range=agent_specification.clip_range,
            ent_coef=agent_specification.ent_coef,
            vf_coef=agent_specification.ent_coef,
            policy_kwargs={
                "features_extractor_class": FeaturesExtractor,
                "features_extractor_kwargs": {
                    "input_dim_features_extractor": env_specification.n_features,
                    "gconv_type": agent_specification.gconv_type,
                    "graph_pooling": agent_specification.graph_pooling,
                    "freeze_graph": agent_specification.freeze_graph,
                    "graph_has_relu": agent_specification.graph_has_relu,
                    "device": agent_specification.device,
                    "max_n_nodes": env_specification.max_n_nodes,
                    "n_mlp_layers_features_extractor": agent_specification.n_mlp_layers_features_extractor,
                    "n_layers_features_extractor": agent_specification.n_layers_features_extractor,
                    "hidden_dim_features_extractor": agent_specification.hidden_dim_features_extractor,
                    "n_attention_heads": agent_specification.n_attention_heads,
                },
                "optimizer_class": agent_specification.optimizer_class,
                "add_boolean": env_specification.add_boolean,
                "mlp_act": agent_specification.mlp_act,
                "_device": agent_specification.device,
                "input_dim_features_extractor": env_specification.n_features,
                "max_n_nodes": env_specification.max_n_nodes,
                "max_n_jobs": env_specification.max_n_jobs,
                "n_layers_features_extractor": agent_specification.n_layers_features_extractor,
                "hidden_dim_features_extractor": agent_specification.hidden_dim_features_extractor,
                "n_mlp_layers_actor": agent_specification.n_mlp_layers_actor,
                "hidden_dim_actor": agent_specification.hidden_dim_actor,
                "n_mlp_layers_critic": agent_specification.n_mlp_layers_critic,
                "hidden_dim_critic": agent_specification.hidden_dim_critic,
            },
            verbose=2,
            device=agent_specification.device,
        )
        self.n_workers = agent_specification.n_workers
        self.device = agent_specification.device
        self.env_specification = env_specification

    def save(self, path):
        """Saving an agent corresponds to saving his model and a few args to specify how the model is working"""
        self.model.save(path)
        with open(path + ".pickle", "wb") as f:
            pickle.dump({"env_specification": self.env_specification, "n_workers": self.n_workers, "device": self.device}, f)

    @classmethod
    def load(cls, path):
        """Loading an agent corresponds to loading his model and a few args to specify how the model is working"""
        with open(path + ".pickle", "rb") as f:
            kwargs = pickle.load(f)
        agent = cls(env_specification=kwargs["env_specification"], model=PPO.load(path))
        agent.n_workers = kwargs["n_workers"]
        agent.device = kwargs["device"]
        return agent

    def train(
        self,
        problem_description,
        training_specification,
    ):
        # First setup callbacks during training
        validation_callback = ValidationCallback(
            problem_description=problem_description,
            env_specification=self.env_specification,
            n_workers=self.n_workers,
            device=self.device,
            n_validation_env=training_specification.n_validation_env,
            display_env=training_specification.display_env,
            path=training_specification.path,
            custom_name=training_specification.custom_heuristic_name,
            max_n_jobs=self.env_specification.max_n_jobs,
            max_n_machines=self.env_specification.max_n_machines,
            max_time_ortools=training_specification.max_time_ortools,
            scaling_constant_ortools=training_specification.scaling_constant_ortools,
            ortools_strategy=training_specification.ortools_strategy,
        )
        event_callback = EveryNTimesteps(n_steps=training_specification.validation_freq, callback=validation_callback)

        # Creating the vectorized environments
        vec_env = DummyVecEnv(
            env_fns=[lambda: Env(problem_description, self.env_specification) for _ in range(self.n_workers)]
        )
        self.model.set_env(vec_env)

        # Launching training
        self.model.learn(training_specification.total_timesteps, callback=event_callback)

    def predict(self, problem_description):
        # Creating an environment on which we will run the inference
        env = Env(problem_description, self.env_specification)

        # Running the inference loop
        observation = env.reset()
        done = False
        while not done:
            action, _ = self.model.predict(observation, deterministic=True)
            observation, reward, done, info = env.step(action)

        return env.get_solution()
