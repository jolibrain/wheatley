from stable_baselines3.common.callbacks import EveryNTimesteps
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.ppo import PPO
from stable_baselines3.a2c import A2C
import torch

from env.env import Env
from models.agent_callback import TestCallback
from models.policy import Policy
from models.features_extractor import FeaturesExtractor
from problem.problem_description import ProblemDescription

from config import MAX_DURATION


class Agent:
    def __init__(
        self,
        n_epochs=None,
        n_steps_episode=None,
        batch_size=None,
        gamma=None,
        clip_range=None,
        target_kl=None,
        ent_coef=None,
        vf_coef=None,
        lr=None,
        optimizer=None,
        freeze_graph=None,
        input_dim_features_extractor=None,
        gconv_type="gin",
        graph_has_relu=False,
        graph_pooling=None,
        add_force_insert_boolean=False,
        slot_locking=False,
        model=None,
        mlp_act="tanh",
        n_workers=1,
        device=None,
        input_list=None,
        fixed_distrib = False
    ):
        fake_env = Agent._create_fake_env(input_list, add_force_insert_boolean, slot_locking, n_workers,fixed_distrib)
        self.fixed_distrib = fixed_distrib
        if model is not None:
            self.model = model
            self.model.set_env(fake_env)
        else:
            if optimizer.lower() == "adam":
                optimizer_class = torch.optim.Adam
            elif optimizer.lower() == "sgd":
                optimizer_class = torch.optim.SGD
            else:
                raise Exception("Optimizer not recognized")

            self.model = PPO(
                Policy,
                fake_env,
                n_epochs=n_epochs,
                n_steps=n_steps_episode,
                batch_size=batch_size,
                gamma=gamma,
                learning_rate=lr,
                clip_range=clip_range,
                target_kl=target_kl,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                verbose=2,
                policy_kwargs={
                    "features_extractor_class": FeaturesExtractor,
                    "features_extractor_kwargs": {
                        "input_dim_features_extractor": input_dim_features_extractor,
                        "gconv_type": gconv_type,
                        "graph_pooling": graph_pooling,
                        "freeze_graph": freeze_graph,
                        "graph_has_relu": graph_has_relu,
                        "device": device,
                    },
                    "optimizer_class": optimizer_class,
                    "add_boolean": add_force_insert_boolean or slot_locking,
                    "mlp_act": mlp_act,
                    "_device": device,
                    "input_dim_features_extractor": input_dim_features_extractor,
                },
                device=device,
                gae_lambda=1,  # To use same vanilla advantage function
            )
        self.input_list = input_list
        self.add_force_insert_boolean = add_force_insert_boolean
        self.slot_locking = slot_locking
        self.n_workers = n_workers
        self.device = device

    def save(self, path):
        self.model.save(path)

    @classmethod
    def load(cls, path, input_list, add_force_insert_boolean, slot_locking, mlp_act, n_workers, device):
        return cls(
            model=PPO.load(
                path,
                Agent._create_fake_env(input_list, add_force_insert_boolean, slot_locking, n_workers),
                # policy_kwargs={
                #     "add_boolean": add_force_insert_boolean or slot_locking,
                #     "mlp_act": mlp_act,
                #     "device": device,
                # },
                device=device,
            ),
            input_list=input_list,
            add_force_insert_boolean=add_force_insert_boolean,
            slot_locking=slot_locking,
            mlp_act=mlp_act,
        )

    def train(
        self,
        problem_description,
        total_timesteps,
        n_test_env,
        eval_freq,
        normalize_input,
        display_env,
        path,
        fixed_benchmark,
        full_force_insert,
        custom_heuristic_name,
        ortools_strategy,
        keep_same_testing_envs
    ):
        # First setup callbacks during training
        test_callback = TestCallback(
            env=Env(
                problem_description,
                normalize_input,
                self.input_list,
                self.add_force_insert_boolean,
                self.slot_locking,
                full_force_insert,
                self.fixed_distrib
            ),
            n_test_env=n_test_env,
            display_env=display_env,
            path=path,
            fixed_benchmark=fixed_benchmark,
            custom_name=custom_heuristic_name,
            ortools_strategy=ortools_strategy,
            keep_same_testing_envs = keep_same_testing_envs
        )
        event_callback = EveryNTimesteps(n_steps=eval_freq, callback=test_callback)

        # Then launch training
        vec_env = make_vec_env(
            self._get_env_fn(problem_description, normalize_input, full_force_insert),
            n_envs=self.n_workers,
            vec_env_cls=DummyVecEnv,
            #            vec_env_kwargs={"start_method": "spawn"},
        )
        self.model.set_env(vec_env)
        self.model.learn(total_timesteps, callback=event_callback)

    def predict(self, problem_description, normalize_input, full_force_insert):
        env = Env(
            problem_description,
            normalize_input=normalize_input,
            input_list=self.input_list,
            add_force_insert_boolean=self.add_force_insert_boolean,
            slot_locking=self.slot_locking,
            full_force_insert=full_force_insert,
        )
        observation = env.reset()
        done = False
        while not done:
            action, _ = self.model.predict(observation, deterministic=True)
            observation, reward, done, info = env.step(action)
        solution = env.get_solution()
        return solution

    @staticmethod
    def _create_fake_env(input_list, add_force_insert_boolean, slot_locking, n_workers,fixed_distrib):
        def f():
            return Env(
                ProblemDescription(2, 2, MAX_DURATION, "L2D", "Sparse"),
                # Sparse is the only type that can be used in both determinisitic and uncertain case
                normalize_input = True,
                input_list = input_list,
                add_force_insert_boolean = add_force_insert_boolean,
                slot_locking = slot_locking,
                fixed_distrib = fixed_distrib
            )

        return make_vec_env(f, n_workers)

    def _get_env_fn(self, problem_description, normalize_input, full_force_insert):
        def f():
            return Env(
                problem_description,
                normalize_input,
                self.input_list,
                self.add_force_insert_boolean,
                self.slot_locking,
                full_force_insert,
                self.fixed_distrib
            )

        return f
