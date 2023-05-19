#
# Wheatley
# Copyright (c) 2023 Jolibrain
# Authors:
#    Guillaume Infantes <guillaume.infantes@jolibrain.com>
#    Antoine Jacquet <antoine.jacquet@jolibrain.com>
#    Michel Thomazo <thomazo.michel@gmail.com>
#    Emmanuel Benazera <emmanuel.benazera@jolibrain.com>
#
#
# This file is part of Wheatley.
#
# Wheatley is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Wheatley is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Wheatley. If not, see <https://www.gnu.org/licenses/>.
#

import copy
import csv
import sys
import time

import numpy as np
import torch
import tqdm
import visdom

from env.jssp_env import JSSPEnv
from env.psp_env import PSPEnv
from models.custom_agent import CustomAgent
from models.random_agent import RandomAgent
from problem.jssp_description import JSSPDescription
from utils.ortools import get_ortools_makespan as get_ortools_makespan_jssp
from utils.ortools import get_ortools_makespan_psp
from utils.utils import (
    decode_mask,
    generate_problem_durations,
    get_obs,
    obs_as_tensor_add_batch_dim,
    safe_mean,
)


class AgentValidator:
    def __init__(
        self,
        problem_description,
        env_specification,
        device,
        training_specification,
        disable_visdom,
        verbose=2,
    ):
        super().__init__()

        # Parameters
        self.problem_description = problem_description
        if isinstance(problem_description, JSSPDescription):
            self.psp = False
        else:
            self.psp = True

        if self.psp:
            self.env_cls = PSPEnv
        else:
            self.env_cls = JSSPEnv
        if training_specification.validate_on_total_data:
            self.env_specification = copy.deepcopy(env_specification)
            self.env_specification.sample_n_jobs = -1
        else:
            self.env_specification = env_specification
        self.device = device

        self.n_validation_env = training_specification.n_validation_env
        self.fixed_validation = training_specification.fixed_validation
        self.fixed_random_validation = training_specification.fixed_random_validation
        self.vis = visdom.Visdom(
            env=training_specification.display_env,
            log_to_filename="/dev/null",
            offline=disable_visdom,
        )
        self.path = training_specification.path
        self.ortools_strategy = training_specification.ortools_strategy

        self.transition_model_config = problem_description.transition_model_config
        self.reward_model_config = problem_description.reward_model_config

        self.custom_name = training_specification.custom_heuristic_name

        self.max_time_ortools = training_specification.max_time_ortools
        self.scaling_constant_ortools = training_specification.scaling_constant_ortools

        # Comparative agents
        self.random_agent = RandomAgent()
        if self.custom_name != "None":
            self.custom_agent = CustomAgent(
                self.env_specification.max_n_jobs,
                self.env_specification.max_n_machines,
                custom_name.lower(),
            )

        # Inner variables
        if hasattr(self.problem_description, "test_psps"):
            mod = len(self.problem_description.test_psps)
        else:
            mod = 1
        self.validation_envs = [
            self.env_cls(
                self.problem_description,
                self.env_specification,
                i % mod,
                validate=True,
            )
            for i in range(self.n_validation_env)
        ]
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
        self.dpss = []
        self.stabilities = []
        self.monotonies = []
        self.total_timestepss = []
        self.first_callback = True
        self.gantt_rl_img = None
        self.gantt_or_img = None
        self.all_or_tools_makespan = []
        self.all_or_tools_schedule = []
        self.time_to_ortools = []
        self.best_makespan_wheatley = float("inf")
        self.best_makespan_ortools = float("inf")
        self.ortools_env_zero_is_optimal = False

        self.batch_size = training_specification.validation_batch_size

        # Compute OR-Tools solutions once if validations are fixed
        if self.fixed_validation:
            self.fixed_ortools = []
            for i in tqdm.tqdm(
                range(self.n_validation_env), desc="Computing fixed OR-Tools solutions"
            ):
                self.fixed_ortools.append(self._get_ortools_makespan(i))
            print(
                "   optimal solutions:",
                sum([res[2] for res in self.fixed_ortools]),
                " / ",
                self.n_validation_env,
            )

        # Compute random solutions once if validations are fixed
        if self.fixed_random_validation:
            print("Computing fixed random solutions ")
            self.fixed_random = []
            for i in tqdm.tqdm(range(self.n_validation_env), desc="   environment"):
                makespans = []
                for j in tqdm.tqdm(
                    range(self.fixed_random_validation),
                    desc="   instance   ",
                    leave=False,
                ):
                    makespans.append(self._get_random_makespan(i))
                self.fixed_random.append(sum(makespans) / len(makespans))

    def validate(self, agent, alg):
        self._evaluate_agent(agent)
        self._save_if_best_model(agent, alg)
        self._visdom_metrics(agent, alg)
        return True

    def _save_if_best_model(self, agent, alg):
        # cur_ratio = np.mean(
        #     np.array(self.makespans[-4 : len(self.makespans)])
        #     / np.array(self.ortools_makespans[-4 : len(self.ortools_makespans)])
        # )
        cur_ratio = self.makespans[-1] / self.ortools_makespans[-1]
        if cur_ratio <= self.makespan_ratio:
            print("Saving agent", self.path + "agent.pkl")
            agent.save(self.path + "agent.pkl")
            torch.save(alg.optimizer.state_dict(), self.path + "optimizer.pkl")

            self.makespan_ratio = cur_ratio
            print(f"Current ratio : {cur_ratio:.3f}")

    def _get_ortools_makespan(self, i):
        if self.psp:
            return get_ortools_makespan_psp(
                self.validation_envs[i],
                self.max_time_ortools,
                self.scaling_constant_ortools,
                self.ortools_strategy,
            )
        else:
            return get_ortools_makespan_jssp(
                self.validation_envs[i].state.affectations,
                self.validation_envs[i].state.original_durations,
                self.env_specification.n_features,
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

    def save_csv(self, name, makespan, optimal, schedule, sampled_jobs):
        f = open(self.path + name + ".csv", "w")
        writer = csv.writer(f)
        if sampled_jobs is not None:
            writer.writerow(["sampled jobs", sampled_jobs])
        writer.writerow(["makespan", makespan])
        writer.writerow(["optimal", optimal])
        writer.writerow([])
        if hasattr(self.env_specification, "max_n_machines"):
            header = [""]
            for i in range(self.env_specification.max_n_machines):
                header.append("task " + str(i) + " start time")
            writer.writerow(header)
            for i in range(schedule.shape[0]):
                line = ["job " + str(i)] + schedule[i].tolist()
                writer.writerow(line)
        else:  # PSP case
            # schedule is (job_schedule, mode)
            header = []
            for i in range(len(schedule[0])):
                header.append("job " + str(i) + " start")
            writer.writerow(header)
            for i in range(len(schedule[0])):
                line = schedule[0]
                writer.writerow(line)
            writer.writerow([])
            header2 = []
            for i in range(len(schedule[0])):
                header.append("job " + str(i) + " mode")
            writer.writerow(header)
            for i in range(len(schedule[0])):
                line = schedule[1]
                writer.writerow(line)
        f.close()

    def rebatch_obs(self, obs_list):
        obs = {}
        for key in obs_list[0]:
            obs[key] = torch.cat([_obs[key] for _obs in obs_list])
        return obs

    def _evaluate_agent(self, agent):
        mean_makespan = 0
        ortools_mean_makespan = 0
        random_mean_makespan = 0
        custom_mean_makespan = 0
        start_eval = time.time()

        if self.batch_size != 0:
            print("batched predicts...")
            # batch inference
            envs = self.validation_envs
            all_rdata = [env.reset(soft=self.fixed_validation) for env in envs]
            all_obs = [
                agent.obs_as_tensor_add_batch_dim(rdata[0]) for rdata in all_rdata
            ]
            all_masks = decode_mask([rdata[1]["mask"] for rdata in all_rdata])
            while envs:
                all_obs = self.rebatch_obs(all_obs)
                all_actions = []
                for i in range(0, len(envs), self.batch_size):
                    bs = min(self.batch_size, len(envs) - i)
                    actions = agent.predict(
                        get_obs(all_obs, list(range(i, i + bs))),
                        action_masks=all_masks[i : i + bs],
                        deterministic=True,
                    )
                    all_actions += list(actions)
                all_obs = []
                all_masks = []
                todo_envs = []
                for env, action in zip(envs, all_actions):
                    obs, _, done, _, info = env.step(action.long().item())
                    if done:
                        continue
                    all_obs.append(agent.obs_as_tensor_add_batch_dim(obs))
                    todo_envs.append(env)
                    all_masks.append(decode_mask(info["mask"]))
                envs = todo_envs
            print("...done")

        for i in tqdm.tqdm(range(self.n_validation_env), desc="   evaluating         "):
            if self.batch_size == 0:
                obs, info = self.validation_envs[i].reset(soft=self.fixed_validation)
                done = False
                while not done:
                    action_masks = decode_mask(info["mask"])
                    obs = agent.obs_as_tensor_add_batch_dim(obs)
                    action = agent.predict(
                        obs, deterministic=True, action_masks=action_masks
                    )
                    obs, reward, done, _, info = self.validation_envs[i].step(
                        action.long().item()
                    )
            solution = self.validation_envs[i].get_solution()
            schedule = solution.schedule
            makespan = solution.get_makespan()

            if i == 0:
                self.gantt_rl_img = self.validation_envs[i].render_solution(schedule)

            if makespan < self.best_makespan_wheatley:
                self.best_makespan_wheatley = makespan
                self.save_csv(
                    "wheatley",
                    makespan,
                    "unknown",
                    schedule,
                    self.validation_envs[i].sampled_jobs,
                )

            mean_makespan += makespan / self.n_validation_env

            if self.fixed_validation:
                or_tools_makespan, or_tools_schedule, optimal = self.fixed_ortools[i]
            else:
                (
                    or_tools_makespan,
                    or_tools_schedule,
                    optimal,
                ) = self._get_ortools_makespan(i)

            if i == 0:
                self.gantt_or_img = self.validation_envs[i].render_solution(
                    or_tools_schedule, scaling=1.0
                )
                self.ortools_env_zero_is_optimal = optimal

            if or_tools_makespan < self.best_makespan_ortools:
                self.best_makespan_ortools = or_tools_makespan
                self.save_csv(
                    "ortools",
                    or_tools_makespan,
                    optimal,
                    or_tools_schedule,
                    self.validation_envs[i].sampled_jobs,
                )

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
                                transition_model_config=self.validation_envs[
                                    i
                                ].transition_model_config,
                                reward_model_config=self.validation_envs[
                                    i
                                ].reward_model_config,
                                affectations=self.validation_envs[
                                    i
                                ].transition_model.affectations,
                                durations=self.validation_envs[
                                    i
                                ].transition_model.durations,
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
        print("--- eval time=", time.time() - start_eval, "  ---")
        self.makespans.append(mean_makespan)
        self.ortools_makespans.append(ortools_mean_makespan)
        self.random_makespans.append(random_mean_makespan)
        self.custom_makespans.append(custom_mean_makespan)

    def _visdom_metrics(self, agent, alg):
        commandline = " ".join(sys.argv)
        html = f"""
            <div style="padding: 5px">
                <h4>Total actions: {alg.global_step}</h4>
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
            opts["linecolor"] = np.array(
                [[31, 119, 180], [255, 127, 14], [44, 160, 44], [255, 0, 0]]
            )
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
        self.vis.line(
            X=X, Y=np.array(self.time_to_ortools), win="time_to_ortools", opts=opts
        )

        if self.first_callback:
            self.first_callback = False
            return

        self.entropy_losses.append(
            alg.ent_coef * alg.logger.name_to_value["train/entropy_loss"]
        )
        self.policy_gradient_losses.append(
            alg.logger.name_to_value["train/policy_gradient_loss"]
        )
        self.value_losses.append(
            alg.vf_coef * alg.logger.name_to_value["train/value_loss"]
        )
        self.losses.append(alg.logger.name_to_value["train/loss"])
        self.approx_kls.append(alg.logger.name_to_value["train/approx_kl"])
        self.clip_fractions.append(alg.logger.name_to_value["train/clip_fraction"])
        self.explained_variances.append(
            alg.logger.name_to_value["train/explained_variance"]
        )
        self.clip_ranges.append(alg.logger.name_to_value["train/clip_range"])
        # Recreate last features by hand, since they are erased
        self.ep_rew_means.append(
            safe_mean([ep_info["r"] for ep_info in alg.ep_info_buffer])
        )
        self.ep_len_means.append(
            safe_mean([ep_info["l"] for ep_info in alg.ep_info_buffer])
        )
        self.fpss.append(int(alg.global_step / (time.time() - alg.start_time)))
        self.dpss.append(
            int(
                alg.n_epochs
                * alg.num_envs
                * alg.num_steps
                / (time.time() - alg.start_time)
            )
        )
        self.total_timestepss.append(alg.global_step)
        self.stabilities.append(alg.logger.name_to_value["train/ratio_stability"])
        self.monotonies.append(alg.logger.name_to_value["train/ratio_monotony"])

        X = list(range(1, len(self.losses) + 1))
        Y_list = [
            self.losses,
            self.value_losses,
            self.policy_gradient_losses,
            self.entropy_losses,
        ]
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
            "updates_per_second": self.dpss,
            "total_timesteps": self.total_timestepss,
            "ratio_stability": self.stabilities,
            "ratio_monotony": self.monotonies,
        }
        for title, data in charts.items():
            self.vis.line(X=X, Y=data, win=title, opts={"title": title})

        if self.gantt_rl_img is not None:
            self.vis.image(
                self.gantt_rl_img,
                opts={"caption": "Gantt RL schedule"},
                win="rl_schedule",
            )
        if self.gantt_or_img is not None:
            if self.ortools_env_zero_is_optimal:
                opts = {"caption": "Gantt OR-Tools schedule (OPTIMAL)"}
            else:
                opts = {"caption": "Gantt OR-Tools schedule (not optimal)"}
            self.vis.image(
                self.gantt_or_img,
                opts=opts,
                win="or_schedule",
            )
