import pickle
import time
import sys
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean
import visdom
import csv
import copy

from env.env import Env
from models.custom_agent import CustomAgent
from models.random_agent import RandomAgent
from problem.problem_description import ProblemDescription
from utils.utils_testing import get_ortools_makespan
from utils.utils import generate_problem_durations
from sb3_contrib.common.maskable.utils import get_action_masks


class ValidationCallback(BaseCallback):
    def __init__(
        self,
        problem_description,
        env_specification,
        n_workers,
        device,
        n_validation_env,
        fixed_validation,
        fixed_random_validation,
        validation_batch_size,
        display_env,
        path,
        custom_name,
        max_n_jobs,
        max_n_machines,
        max_time_ortools,
        scaling_constant_ortools,
        ortools_strategy="pessimistic",
        validate_on_total_data=False,
        verbose=2,
    ):
        super(ValidationCallback, self).__init__(verbose=verbose)

        # Parameters
        self.problem_description = problem_description
        if validate_on_total_data:
            self.env_specification = copy.deepcopy(env_specification)
            self.env_specification.sample_n_jobs = -1
        else:
            self.env_specification = env_specification
        self.n_workers = n_workers
        self.device = device

        self.n_validation_env = n_validation_env
        self.fixed_validation = fixed_validation
        self.fixed_random_validation = fixed_random_validation
        self.validation_batch_size = validation_batch_size
        self.vis = visdom.Visdom(env=display_env)
        self.path = path
        self.ortools_strategy = ortools_strategy

        self.n_jobs = problem_description.n_jobs
        self.n_machines = problem_description.n_machines
        self.transition_model_config = problem_description.transition_model_config
        self.reward_model_config = problem_description.reward_model_config

        self.custom_name = custom_name

        self.max_n_jobs = max_n_jobs
        self.max_n_machines = max_n_machines
        self.max_time_ortools = max_time_ortools
        self.scaling_constant_ortools = scaling_constant_ortools

        # Comparative agents
        self.random_agent = RandomAgent(self.max_n_jobs, self.max_n_machines)
        if custom_name != "None":
            self.custom_agent = CustomAgent(self.max_n_jobs, self.max_n_machines, custom_name.lower())

        # Inner variables
        self.validation_envs = [Env(self.problem_description, self.env_specification) for _ in range(self.n_validation_env)]
        self.makespan_ratio = 1000
        self.makespans = []
        self.ortools_makespans = []
        self.random_makespans = []
        self.custom_makespans = []
        self.entropy_losses = []
        self.policy_gradient_losses = []
        self.value_losses = []
        self.losses = []
        self.approx_kls = []
        self.clip_fractions = []
        self.explained_variances = []
        self.clip_ranges = []
        self.ep_len_means = []
        self.ep_rew_means = []
        self.fpss = []
        self.total_timestepss = []
        self.first_callback = True
        self.gantt_rl_img = None
        self.gantt_or_img = None
        self.all_or_tools_makespan = []
        self.all_or_tools_schedule = []
        self.time_to_ortools = []
        self.best_makespan_wheatley = float("inf")
        self.best_makespan_ortools = float("inf")

        # Compute OR-Tools solutions once if validations are fixed
        if fixed_validation:
            print("Computing fixed OR-Tools solutions", end="", flush=True)
            self.fixed_ortools = []
            for i in range(self.n_validation_env):
                self.fixed_ortools.append(self._get_ortools_makespan(i))
                print(".", end="", flush=True)
            print()

        # Compute random solutions once if validations are fixed
        if fixed_random_validation:
            print("Computing fixed random solutions", end="", flush=True)
            self.fixed_random = []
            for i in range(self.n_validation_env):
                makespans = [self._get_random_makespan(i) for n in range(fixed_validation)]
                self.fixed_random.append(sum(makespans) / len(makespans))
                print(".", end="", flush=True)
            print()

    def _on_step(self):
        self._evaluate_agent()
        self._save_if_best_model()
        self._visdom_metrics()
        print(self.path)
        return True

    def _save_if_best_model(self):
        cur_ratio = np.mean(
            np.array(self.makespans[-4 : len(self.makespans)])
            / np.array(self.ortools_makespans[-4 : len(self.ortools_makespans)])
        )
        if cur_ratio <= self.makespan_ratio:
            self.model.save(self.path)

            # EVIL QUICK FIX OF DEATH
            # The best thing to do would be to specify a AgentTrainer, responsible of training the agent and printing
            # callbacks. This way, we could just do agent.save(path). This needs to rebuild the structure of the training.
            with open(self.path + ".pickle", "wb") as f:
                pickle.dump(
                    {"env_specification": self.env_specification, "n_workers": self.n_workers, "device": self.device}, f
                )

            self.makespan_ratio = cur_ratio
            print("Saving model")
            print(f"Current ratio : {cur_ratio:.3f}")

    def _get_ortools_makespan(self, i):
        return get_ortools_makespan(
            self.validation_envs[i].state.affectations,
            self.validation_envs[i].state.original_durations,
            self.max_time_ortools,
            self.scaling_constant_ortools,
            self.ortools_strategy,
        )

    def _get_random_makespan(self, i):
        return self.random_agent.predict(self.validation_envs[i]).get_makespan()

    # transform list of dicts to dict of lists
    def _list_to_dict(self, batch_list):
        batch_dict = {}
        for obs in batch_list:
            for key, value in obs.items():
                batch_dict.setdefault(key, []).append(value)
        return batch_dict

    def save_csv(self, name, makespan, schedule, sampled_jobs):
        f = open(self.path + "." + name + ".csv", "w")
        writer = csv.writer(f)
        if sampled_jobs is not None:
            writer.writerow(["sampled jobs", sampled_jobs])
        writer.writerow(["makespan", makespan])
        writer.writerow([])
        header = [""]
        for i in range(self.max_n_machines):
            header.append("task " + str(i) + " start time")
        writer.writerow(header)
        for i in range(schedule.shape[0]):
            line = ["job " + str(i)] + schedule[i].tolist()
            writer.writerow(line)
        f.close()

    def _evaluate_agent(self):
        # compute the solutions in parallel if we use batch_size
        batch_size = self.validation_batch_size
        if batch_size:
            envs = self.validation_envs
            all_obs = [env.reset(soft=self.fixed_validation) for env in envs]
            while envs:
                all_masks = [get_action_masks(env) for env in envs]
                all_actions = []
                for i in range(0, len(envs), batch_size):
                    actions, _ = self.model.predict(
                        self._list_to_dict(all_obs[i : i + batch_size]),
                        action_masks=all_masks[i : i + batch_size],
                        deterministic=True,
                    )
                    all_actions += list(actions)
                all_obs = []
                todo_envs = []
                for env, action in zip(envs, all_actions):
                    obs, reward, done, info = env.step(action)
                    if done:
                        continue
                    all_obs.append(obs)
                    todo_envs.append(env)
                envs = todo_envs

        mean_makespan = 0
        ortools_mean_makespan = 0
        random_mean_makespan = 0
        custom_mean_makespan = 0
        for i in range(self.n_validation_env):
            if not batch_size:
                obs = self.validation_envs[i].reset(soft=self.fixed_validation)
                done = False
                while not done:
                    action_masks = get_action_masks(self.validation_envs[i])
                    action, _ = self.model.predict(obs, deterministic=True, action_masks=action_masks)
                    obs, reward, done, info = self.validation_envs[i].step(action)
            solution = self.validation_envs[i].get_solution()
            schedule = solution.schedule
            makespan = solution.get_makespan()

            if i == 0:
                self.gantt_rl_img = self.validation_envs[i].render_solution(schedule)

            if makespan < self.best_makespan_wheatley:
                self.best_makespan_wheatley = makespan
                self.save_csv("wheatley", makespan, schedule, self.validation_envs[i].sampled_jobs)

            mean_makespan += makespan / self.n_validation_env

            if self.fixed_validation:
                or_tools_makespan, or_tools_schedule = self.fixed_ortools[i]
            else:
                or_tools_makespan, or_tools_schedule = self._get_ortools_makespan(i)

            if i == 0:
                self.gantt_or_img = self.validation_envs[i].render_solution(or_tools_schedule, scaling=1.0)

            if or_tools_makespan < self.best_makespan_ortools:
                self.best_makespan_ortools = or_tools_makespan
                self.save_csv("ortools", or_tools_makespan.item(), or_tools_schedule, self.validation_envs[i].sampled_jobs)

            ortools_mean_makespan += or_tools_makespan / self.n_validation_env

            if self.fixed_random_validation:
                random_makespan = self.fixed_random[i]
            else:
                random_makespan = self._get_random_makespan(i)
            random_mean_makespan += random_makespan / self.n_validation_env

            if self.custom_name != "None":
                custom_mean_makespan += (
                    np.max(
                        self.custom_agent.predict(
                            ProblemDescription(
                                transition_model_config=self.validation_envs[i].transition_model_config,
                                reward_model_config=self.validation_envs[i].reward_model_config,
                                affectations=self.validation_envs[i].transition_model.affectations,
                                durations=self.validation_envs[i].transition_model.durations,
                                n_jobs=self.validation_envs[i].n_jobs,
                                n_machines=self.validation_envs[i].n_machines,
                            ),
                            True,
                            None,
                        ).get_makespan()
                    )
                    / self.n_validation_env
                )
        print("--- mean_makespan=", mean_makespan, " ---")
        self.makespans.append(mean_makespan)
        self.ortools_makespans.append(ortools_mean_makespan)
        self.random_makespans.append(random_mean_makespan)
        self.custom_makespans.append(custom_mean_makespan)

    def _visdom_metrics(self):
        commandline = " ".join(sys.argv)
        html = f"""
            <div style="padding: 5px">
                <h4>Total actions: {self.model.num_timesteps}</h4>
                <code>{commandline}</code>
            </div>
        """
        self.vis.text(html, win="html", opts={"width": 372, "height": 336})

        X = list(range(len(self.makespans)))
        Y_list = [self.makespans, self.random_makespans, self.ortools_makespans]
        opts = {
            "legend": ["PPO", "Random", "OR-tools"],
            "linecolor": np.array([[31, 119, 180], [255, 127, 14], [44, 160, 44]]),
        }
        Y2_list = [
            np.array(self.makespans) / np.array(self.ortools_makespans),
            np.array(self.random_makespans) / np.array(self.ortools_makespans),
        ]
        opts2 = {
            "legend": ["PPO / OR-tools", "Random / OR-tools"],
            "linecolor": np.array([[31, 119, 180], [255, 127, 14]]),
        }
        if self.custom_name != "None":
            Y_list.append(self.custom_makespans)
            Y2_list.append(self.custom_makespans / np.array(self.ortools_makespans))
            opts["legend"].append(self.custom_name)
            opts["linecolor"] = np.array([[31, 119, 180], [255, 127, 14], [44, 160, 44], [255, 0, 0]])
            opts2["legend"].append(self.custom_name + " / OR-tools")
            opts2["linecolor"] = np.array([[31, 119, 180], [255, 127, 14], [255, 0, 0]])
        self.vis.line(X=X, Y=np.array(Y_list).T, win="validation_makespan", opts=opts)
        # self.vis.line(X=X, Y=np.stack(Y2_list).T, win="validation_makespan_ratio", opts=opts2)

        # ratio to OR-tools
        opts = {"title": "PPO / OR-tools"}
        ratio_to_ortools = np.array(self.makespans) / np.array(self.ortools_makespans)
        self.vis.line(X=X, Y=ratio_to_ortools, win="ratio_to_ortools", opts=opts)

        # distance to OR-tools
        opts = {"title": "Distance to OR-tools"}
        dist_to_ortools = np.array(self.makespans) - np.array(self.ortools_makespans)
        self.vis.line(X=X, Y=dist_to_ortools, win="dist_to_ortools", opts=opts)

        # time to OR-tools
        wins = 0
        count = min(len(self.makespans), 100)
        for i in range(1, count + 1):
            ppo = self.makespans[-i]
            ortools = self.ortools_makespans[-i]
            if ppo <= ortools or np.isclose(ppo, ortools):
                wins += 1
        pct = 100 * wins / count
        self.time_to_ortools.append(pct)
        opts = {"title": "Time to OR-tools %"}
        self.vis.line(X=X, Y=np.array(self.time_to_ortools), win="time_to_ortools", opts=opts)

        if self.first_callback:
            self.first_callback = False
            return

        self.entropy_losses.append(self.model.ent_coef * self.model.logger.name_to_value["train/entropy_loss"])
        self.policy_gradient_losses.append(self.model.logger.name_to_value["train/policy_gradient_loss"])
        self.value_losses.append(self.model.vf_coef * self.model.logger.name_to_value["train/value_loss"])
        self.losses.append(self.model.logger.name_to_value["train/loss"])
        self.approx_kls.append(self.model.logger.name_to_value["train/approx_kl"])
        self.clip_fractions.append(self.model.logger.name_to_value["train/clip_fraction"])
        self.explained_variances.append(self.model.logger.name_to_value["train/explained_variance"])
        self.clip_ranges.append(self.model.logger.name_to_value["train/clip_range"])
        # Recreate last features by hand, since they are erased
        self.ep_rew_means.append(safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer]))
        self.ep_len_means.append(safe_mean([ep_info["l"] for ep_info in self.model.ep_info_buffer]))
        self.fpss.append(int(self.model.num_timesteps / (time.time() - self.model.start_time)))
        self.total_timestepss.append(self.model.num_timesteps)

        X = list(range(1, len(self.losses) + 1))
        Y_list = [self.losses, self.value_losses, self.policy_gradient_losses, self.entropy_losses]
        opts = {
            "legend": ["loss", "value_loss", "policy_gradient_loss", "entropy_loss"],
        }
        self.vis.line(X=X, Y=np.array(Y_list).T, win="losses", opts=opts)

        charts = {
            "entropy_loss": self.entropy_losses,
            "policy_gradient_loss": self.policy_gradient_losses,
            "value_loss": self.value_losses,
            "loss": self.losses,
            "approx_kl": self.approx_kls,
            "clip_fraction": self.clip_fractions,
            "explained_variance": self.explained_variances,
            "clip_range": self.clip_ranges,
            "ep_len_mean": self.ep_len_means,
            "ep_rew_mean": self.ep_rew_means,
            "actions_per_second": self.fpss,
            "total_timesteps": self.total_timestepss,
        }
        for title, data in charts.items():
            self.vis.line(X=X, Y=data, win=title, opts={"title": title})

        if self.gantt_rl_img is not None:
            self.vis.image(self.gantt_rl_img, opts={"caption": "Gantt RL schedule"}, win="rl_schedule")
        if self.gantt_or_img is not None:
            self.vis.image(self.gantt_or_img, opts={"caption": "Gantt OR-Tools schedule"}, win="or_schedule")
