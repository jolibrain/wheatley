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
import pickle
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
        # self.ortools_strategies = [training_specification.ortools_strategy, "realistic"]
        self.ortools_strategies = training_specification.ortools_strategy

        self.transition_model_config = problem_description.transition_model_config
        self.reward_model_config = problem_description.reward_model_config

        self.custom_names = training_specification.custom_heuristic_names

        self.max_time_ortools = training_specification.max_time_ortools
        self.scaling_constant_ortools = training_specification.scaling_constant_ortools

        # Comparative agents
        self.random_agent = RandomAgent()
        self.custom_agents = [
            CustomAgent(rule, stochasticity_strategy="averagistic")
            for rule in self.custom_names
        ]

        # Inner variables
        if hasattr(self.problem_description, "test_psps"):
            n_test_pb = len(self.problem_description.test_psps)

            if (
                n_test_pb == self.n_validation_env and self.fixed_random_validation == 0
            ):  # if possible, one env per test, good for deterministic
                aff = [[i] for i in range(self.n_validation_env)]
            else:  # TODO: maybe use fixed random validation as number of sample of tests
                aff = [
                    list(range(len(self.problem_description.test_psps)))
                ] * self.fixed_random_validation
                self.n_validation_env = len(aff)

        else:
            aff = [[0]] * self.n_validation_env

        self.validation_envs = [
            self.env_cls(
                self.problem_description,
                self.env_specification,
                aff[i],
                validate=True,
            )
            for i in range(self.n_validation_env)
        ]
        self.makespan_ratio = 1000
        self.makespans = []
        self.ortools_makespans = {
            ortools_strategy: [] for ortools_strategy in self.ortools_strategies
        }
        self.random_makespans = []
        self.custom_makespans = {agent.rule: [] for agent in self.custom_agents}
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
        self.gantt_mopnr_img = None
        self.all_or_tools_makespan = []
        self.all_or_tools_schedule = []
        self.time_to_ortools = []
        self.best_makespan_wheatley = [float("inf")] * self.n_validation_env
        self.best_makespan_ortools = [float("inf")] * self.n_validation_env
        self.ortools_env_zero_is_optimal = False

        self.batch_size = training_specification.validation_batch_size

        # Compute OR-Tools solutions once if validations are fixed
        if self.fixed_validation:
            self.fixed_ortools = {
                ortools_strategy: [] for ortools_strategy in self.ortools_strategies
            }
            for i in tqdm.tqdm(
                range(self.n_validation_env), desc="Computing fixed OR-Tools solutions"
            ):
                for ortools_strategy in self.ortools_strategies:
                    self.fixed_ortools[ortools_strategy].append(
                        self._get_ortools_makespan(i, ortools_strategy)
                    )

            print(
                f"   optimal solutions ({self.default_ortools_strategy}):",
                sum(
                    [
                        res[2]
                        for res in self.fixed_ortools[self.default_ortools_strategy]
                    ]
                ),
                " / ",
                self.n_validation_env,
            )

            self.fixed_custom_solutions = dict()
            for agent in self.custom_agents:
                self.fixed_custom_solutions[agent.rule] = []
                for i in tqdm.tqdm(
                    range(self.n_validation_env),
                    desc=f"Computing fixed {agent.rule} solutions",
                ):
                    solution = agent.predict(
                        self.validation_envs[i].state.original_durations,
                        self.validation_envs[i].state.affectations,
                    )
                    self.fixed_custom_solutions[agent.rule].append(
                        solution.get_makespan()
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
        self._visdom_metrics(agent, alg)
        self._save_if_best_model(agent, alg)
        return True

    def _save_if_best_model(self, agent, alg):
        # cur_ratio = np.mean(
        #     np.array(self.makespans[-4 : len(self.makespans)])
        #     / np.array(self.ortools_makespans[-4 : len(self.ortools_makespans)])
        # )

        cur_ratio = (
            self.makespans[-1]
            / self.ortools_makespans[self.default_ortools_strategy][-1]
        )
        if cur_ratio <= self.makespan_ratio:
            print("Saving agent", self.path + "agent.pkl")
            agent.save(self.path + "agent.pkl")
            torch.save(alg.optimizer.state_dict(), self.path + "optimizer.pkl")
            self.save_state(self.path + "validator.pkl")

            self.makespan_ratio = cur_ratio
            print(f"Current ratio : {cur_ratio:.3f}")

    def _get_ortools_makespan(self, i: int, ortools_strategy: str):
        if self.psp:
            return get_ortools_makespan_psp(
                self.validation_envs[i],
                self.max_time_ortools,
                self.scaling_constant_ortools,
                ortools_strategy,
            )
        else:
            return get_ortools_makespan_jssp(
                self.validation_envs[i].state.affectations,
                self.validation_envs[i].state.original_durations,
                self.env_specification.n_features,
                self.max_time_ortools,
                self.scaling_constant_ortools,
                ortools_strategy,
            )

    def _get_random_makespan(self, i):
        sol = self.random_agent.predict(self.validation_envs[i])
        if sol is None:
            return self.validation_envs[i].state.undoable_makespan
        return sol.get_makespan()

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
        ortools_mean_makespan = {
            ortools_strategy: 0 for ortools_strategy in self.ortools_strategies
        }
        random_mean_makespan = 0
        custom_mean_makespan = {agent.rule: 0 for agent in self.custom_agents}
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
            if solution is not None:
                schedule = solution.schedule
                makespan = solution.get_makespan()

                if i == 0:
                    self.gantt_rl_img = self.validation_envs[i].render_solution(
                        schedule
                    )

                if makespan < self.best_makespan_wheatley[i]:
                    self.best_makespan_wheatley[i] = makespan
                    self.save_csv(
                        f"wheatley_{i}",
                        makespan,
                        "unknown",
                        schedule,
                        self.validation_envs[i].sampled_jobs,
                    )

                mean_makespan += makespan / self.n_validation_env
            else:
                schedule = None
                state = self.validation_envs[i].state
                mean_makespan += state.undoable_makespan / self.n_validation_env
                self.gantt_rl_img = self.validation_envs[i].render_fail()

            for ortools_strategy in self.ortools_strategies:
                if self.fixed_validation:
                    (
                        or_tools_makespan,
                        or_tools_schedule,
                        optimal,
                    ) = self.fixed_ortools[ortools_strategy][i]
                else:
                    (
                        or_tools_makespan,
                        or_tools_schedule,
                        optimal,
                    ) = self._get_ortools_makespan(i, ortools_strategy)

                if i == 0 and ortools_strategy == self.default_ortools_strategy:
                    self.gantt_or_img = self.validation_envs[i].render_solution(
                        or_tools_schedule, scaling=1.0
                    )
                    self.ortools_env_zero_is_optimal = optimal

                if (
                    ortools_strategy == self.default_ortools_strategy
                    and or_tools_makespan < self.best_makespan_ortools[i]
                ):
                    self.best_makespan_ortools[i] = or_tools_makespan
                    self.save_csv(
                        f"ortools_{i}",
                        or_tools_makespan,
                        optimal,
                        or_tools_schedule,
                        self.validation_envs[i].sampled_jobs,
                    )

                ortools_mean_makespan[ortools_strategy] += (
                    or_tools_makespan / self.n_validation_env
                )

            if self.fixed_random_validation:
                random_makespan = self.fixed_random[i]
            else:
                random_makespan = self._get_random_makespan(i)
            random_mean_makespan += random_makespan / self.n_validation_env

            for custom_agent in self.custom_agents:
                name = custom_agent.rule

                if self.fixed_validation:
                    makespan = self.fixed_custom_solutions[name][i]
                else:
                    solution = custom_agent.predict(
                        self.validation_envs[i].state.original_durations,
                        self.validation_envs[i].state.affectations,
                    )
                    makespan = solution.get_makespan()
                    if custom_agent.rule == "MOPNR" and i == 0:
                        self.gantt_mopnr_img = self.validation_envs[i].render_solution(
                            solution.schedule
                        )

                custom_mean_makespan[name] += makespan / self.n_validation_env

        print("--- mean_makespan=", mean_makespan, " ---")
        print("--- eval time=", time.time() - start_eval, "  ---")
        self.makespans.append(mean_makespan)
        for ortools_strategy in self.ortools_strategies:
            self.ortools_makespans[ortools_strategy].append(
                ortools_mean_makespan[ortools_strategy]
            )
        self.random_makespans.append(random_mean_makespan)
        for custom_agent in self.custom_agents:
            self.custom_makespans[custom_agent.rule].append(
                custom_mean_makespan[custom_agent.rule]
            )

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
        Y_list = [self.makespans, self.random_makespans]
        opts = {
            "title": "Validation makespan",
            "legend": ["PPO", "Random"],
            "linecolor": [[31, 119, 180], [255, 127, 14]],
        }
        Y2_list = [
            np.array(self.makespans)
            / np.array(self.ortools_makespans[self.default_ortools_strategy]),
            np.array(self.random_makespans)
            / np.array(self.ortools_makespans[self.default_ortools_strategy]),
        ]
        opts2 = {
            "legend": ["PPO / OR-tools", "Random / OR-tools"],
            "linecolor": [[31, 119, 180], [255, 127, 14]],
        }

        # OR-Tools plots
        ortools_colors = [
            [44, 160, 44],
            [214, 39, 40],
            [148, 103, 189],
            [140, 86, 75],
            [227, 119, 194],
            [127, 127, 127],
        ]
        for ortools_strategy_id, ortools_strategy in enumerate(self.ortools_strategies):
            Y_list.append(self.ortools_makespans[ortools_strategy])
            opts["legend"].append("OR-Tools - " + ortools_strategy)
            opts["linecolor"].append(ortools_colors[ortools_strategy_id])

        # Custom agent plots
        agent_colors = np.array(
            [
                [255, 0, 0],
                [200, 0, 0],
                [150, 0, 0],
                [150, 50, 0],
                [127, 127, 0],
            ]
        )
        for custom_agent_id, custom_agent in enumerate(self.custom_agents):
            name = custom_agent.rule
            Y = np.array(self.custom_makespans[name])
            Y_list.append(Y)
            Y2_list.append(
                Y / np.array(self.ortools_makespans[self.default_ortools_strategy])
            )

            opts["linecolor"].append(agent_colors[custom_agent_id])
            opts["legend"].append(name)

            opts2["legend"].append(name + " / OR-tools")
            opts2["linecolor"].append(agent_colors[custom_agent_id])

        opts["linecolor"] = np.array(opts["linecolor"])
        self.vis.line(X=X, Y=np.array(Y_list).T, win="validation_makespan", opts=opts)
        # opts2["linecolor"] = np.array(opts2["linecolor"])
        # self.vis.line(X=X, Y=np.stack(Y2_list).T, win="validation_makespan_ratio", opts=opts2)

        # ratio to OR-tools
        opts = {
            "title": "PPO / OR-tools",
            "legend": ["PPO/OR-tools", "Min PPO/OR-tools"],
        }
        ratio_to_ortools = np.array(self.makespans) / np.array(
            self.ortools_makespans[self.default_ortools_strategy]
        )
        min_ratio_to_ortools = np.minimum.accumulate(ratio_to_ortools)
        self.vis.line(
            X=X,
            Y=np.stack([ratio_to_ortools, min_ratio_to_ortools], axis=1),
            win="ratio_to_ortools",
            opts=opts,
        )

        # distance to OR-tools
        opts = {"title": "Distance to OR-tools"}
        dist_to_ortools = np.array(self.makespans) - np.array(
            self.ortools_makespans[self.default_ortools_strategy]
        )
        self.vis.line(X=X, Y=dist_to_ortools, win="dist_to_ortools", opts=opts)

        # time to OR-tools
        wins = 0
        count = min(len(self.makespans), 100)
        for i in range(1, count + 1):
            ppo = self.makespans[-i]
            ortools = self.ortools_makespans[self.default_ortools_strategy][-i]
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

        if self.gantt_mopnr_img is not None:
            self.vis.image(
                self.gantt_mopnr_img,
                opts={"caption": "Gantt MOPNR schedule"},
                win="mopnr_schedule",
            )

    @property
    def default_ortools_strategy(self) -> str:
        if "realistic" in self.ortools_strategies:
            return "realistic"
        elif "averagistic" in self.ortools_strategies:
            return "averagistic"
        else:
            return self.ortools_strategies[0]

    def save_state(self, filepath: str):
        validator_state = {
            "makespan_ratio": self.makespan_ratio,
            "makespans": self.makespans,
            "ortools_makespans": self.ortools_makespans,
            "random_makespans": self.random_makespans,
            "custom_makespans": self.custom_makespans,
            "entropy_losses": self.entropy_losses,
            "policy_gradient_losses": self.policy_gradient_losses,
            "value_losses": self.value_losses,
            "losses": self.losses,
            "approx_kls": self.approx_kls,
            "clip_fractions": self.clip_fractions,
            "explained_variances": self.explained_variances,
            "clip_ranges": self.clip_ranges,
            "ep_len_means": self.ep_len_means,
            "ep_rew_means": self.ep_rew_means,
            "fpss": self.fpss,
            "dpss": self.dpss,
            "stabilities": self.stabilities,
            "monotonies": self.monotonies,
            "total_timestepss": self.total_timestepss,
            "first_callback": self.first_callback,
            "gantt_rl_img": self.gantt_rl_img,
            "gantt_or_img": self.gantt_or_img,
            "gantt_mopnr_img": self.gantt_mopnr_img,
            "all_or_tools_makespan": self.all_or_tools_makespan,
            "all_or_tools_schedule": self.all_or_tools_schedule,
            "time_to_ortools": self.time_to_ortools,
            "best_makespan_wheatley": self.best_makespan_wheatley,
            "best_makespan_ortools": self.best_makespan_ortools,
            "ortools_env_zero_is_optimal": self.ortools_env_zero_is_optimal,
        }
        with open(filepath, "wb") as f:
            pickle.dump(validator_state, f)

    def reload_state(self, filepath: str) -> "AgentValidator":
        with open(filepath, "rb") as f:
            validator_state = pickle.load(f)

        self.makespan_ratio = validator_state["makespan_ratio"]
        self.makespans = validator_state["makespans"]
        self.ortools_makespans = validator_state["ortools_makespans"]
        self.random_makespans = validator_state["random_makespans"]
        self.custom_makespans = validator_state["custom_makespans"]
        self.entropy_losses = validator_state["entropy_losses"]
        self.policy_gradient_losses = validator_state["policy_gradient_losses"]
        self.value_losses = validator_state["value_losses"]
        self.losses = validator_state["losses"]
        self.approx_kls = validator_state["approx_kls"]
        self.clip_fractions = validator_state["clip_fractions"]
        self.explained_variances = validator_state["explained_variances"]
        self.clip_ranges = validator_state["clip_ranges"]
        self.ep_len_means = validator_state["ep_len_means"]
        self.ep_rew_means = validator_state["ep_rew_means"]
        self.fpss = validator_state["fpss"]
        self.dpss = validator_state["dpss"]
        self.stabilities = validator_state["stabilities"]
        self.monotonies = validator_state["monotonies"]
        self.total_timestepss = validator_state["total_timestepss"]
        self.first_callback = validator_state["first_callback"]
        self.gantt_rl_img = validator_state["gantt_rl_img"]
        self.gantt_or_img = validator_state["gantt_or_img"]
        self.gantt_mopnr_img = validator_state["gantt_mopnr_img"]
        self.all_or_tools_makespan = validator_state["all_or_tools_makespan"]
        self.all_or_tools_schedule = validator_state["all_or_tools_schedule"]
        self.time_to_ortools = validator_state["time_to_ortools"]
        self.best_makespan_wheatley = validator_state["best_makespan_wheatley"]
        self.best_makespan_ortools = validator_state["best_makespan_ortools"]
        self.ortools_env_zero_is_optimal = validator_state[
            "ortools_env_zero_is_optimal"
        ]

        return self
