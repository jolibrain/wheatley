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
import os
import pickle
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import visdom
from PIL import Image

from generic.random_agent import RandomAgent
from generic.utils import decode_mask, safe_mean
from jssp.description import Description as JSSPDescription
from jssp.env.env import Env as JSSPEnv
from jssp.models.custom_agent import CustomAgent
from jssp.utils.ortools import get_ortools_makespan as get_ortools_makespan_jssp
from psp.env.env import Env as PSPEnv
from psp.env.genv import GEnv
from psp.utils.ortools import get_ortools_criterion_psp


class AgentValidator:
    def __init__(
        self,
        problem_description,
        env_specification,
        device,
        training_specification,
        disable_visdom,
        validation_envs=None,
        verbose=2,
        graphobs=False,
        compute_ortools=True,
    ):
        super().__init__()

        # Parameters
        self.problem_description = problem_description
        if isinstance(problem_description, JSSPDescription):
            self.psp = False
        else:
            self.psp = True

        self.graphobs = graphobs

        self.compute_ortools = compute_ortools

        if self.psp:
            if self.graphobs:
                self.env_cls = GEnv
            else:
                self.env_cls = PSPEnv
        else:
            self.env_cls = JSSPEnv
        if training_specification.validate_on_total_data:
            self.env_specification = copy.deepcopy(env_specification)
            self.env_specification.sample_n_jobs = -1
        else:
            self.env_specification = env_specification
        self.device = device

        self.display_gantt = training_specification.display_gantt

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
        if self.compute_ortools:
            self.ortools_strategies = training_specification.ortools_strategy

        self.transition_model_config = problem_description.transition_model_config
        self.reward_model_config = problem_description.reward_model_config

        self.custom_names = training_specification.custom_heuristic_names

        if self.compute_ortools:
            self.max_time_ortools = training_specification.max_time_ortools
            self.scaling_constant_ortools = (
                training_specification.scaling_constant_ortools
            )
        self.debug_net = training_specification.debug_net

        # Comparative agents
        self.random_agent = RandomAgent()
        self.custom_agents = [
            CustomAgent(rule, stochasticity_strategy="averagistic")
            for rule in self.custom_names
        ]

        # Inner variables
        if hasattr(self.problem_description, "test_psps"):
            if self.problem_description.unload:
                n_test_pb = len(self.problem_description.test_psps_ids)
            else:
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

        if validation_envs is not None:
            self.validation_envs = validation_envs
        else:
            self.validation_envs = []
            for i in tqdm.tqdm(
                range(self.n_validation_env), desc="Creating validation envs"
            ):
                problem_description = copy.deepcopy(self.problem_description)
                problem_description.rng = np.random.default_rng(
                    self.problem_description.seed + i
                )
                self.validation_envs.append(
                    self.env_cls(
                        problem_description,
                        self.env_specification,
                        aff[i],
                        validate=True,
                        pyg=self.env_specification.pyg,
                    )
                )

        self.criterion_ratio = 1000
        self.criterions = []
        if self.compute_ortools:
            self.ortools_criterions = {
                ortools_strategy: [] for ortools_strategy in self.ortools_strategies
            }
            self.last_ortools_criterions = [0 for _ in range(self.n_validation_env)]
        self.last_ppo_criterions = [0 for _ in range(self.n_validation_env)]
        self.random_criterions = []
        self.custom_criterions = {agent.rule: [] for agent in self.custom_agents}
        self.entropy_losses = []
        self.policy_gradient_losses = []
        self.value_losses = []
        self.losses = []
        self.approx_kls = []
        self.clip_fractions = []
        self.explained_variances = []
        self.return_variances = []
        self.value_variances = []
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
        self.current_scatter_fig = None
        self.all_or_tools_criterion = []
        self.all_or_tools_schedule = []
        if self.compute_ortools:
            self.time_to_ortools = []
            self.best_criterion_ortools = [float("inf")] * self.n_validation_env
            self.ortools_env_zero_is_optimal = False

        self.best_criterion_wheatley = [float("inf")] * self.n_validation_env
        self.variances = {}

        self.batch_size = training_specification.validation_batch_size

        # Compute OR-Tools solutions once if validations are fixed
        if self.fixed_validation:
            if self.compute_ortools:
                self.fixed_ortools = {
                    ortools_strategy: [] for ortools_strategy in self.ortools_strategies
                }
                for i in tqdm.tqdm(
                    range(self.n_validation_env),
                    desc="Computing fixed OR-Tools solutions",
                ):
                    for ortools_strategy in self.ortools_strategies:
                        self.fixed_ortools[ortools_strategy].append(
                            self._get_ortools_criterion(
                                i,
                                ortools_strategy,
                                self.problem_description.reward_model_config,
                            )
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
                        solution.get_criterion()
                    )

        # Compute random solutions once if validations are fixed
        if self.fixed_random_validation:
            print("Computing fixed random solutions ")
            self.fixed_random = []
            for i in tqdm.tqdm(range(self.n_validation_env), desc="   environment"):
                criterions = []
                for j in tqdm.tqdm(
                    range(self.fixed_random_validation),
                    desc="   instance   ",
                    leave=False,
                ):
                    criterions.append(self._get_random_criterion(i))
                self.fixed_random.append(sum(criterions) / len(criterions))

    def validate(self, agent, alg):
        solutions = self._evaluate_agent(agent)
        self._visdom_metrics(agent, alg)
        self._save_if_best_model(agent, alg, solutions)
        return True

    def _save_if_best_model(self, agent, alg, solutions):
        # cur_ratio = np.mean(
        #     np.array(self.criterions[-4 : len(self.criterions)])
        #     / np.array(self.ortools_criterions[-4 : len(self.ortools_criterions)])
        # )

        if self.compute_ortools:
            cur_ratio = (
                self.criterions[-1]
                / self.ortools_criterions[self.default_ortools_strategy][-1]
            )
        else:
            cur_ratio = (
                self.criterions[-1] / self.criterions[0]
                if self.criterions[0] != 0
                else self.criterions[-1]
            )

        if cur_ratio <= self.criterion_ratio:
            print("saving solutions")
            for i, sol in enumerate(solutions):
                if hasattr(self.problem_description, "test_psps"):
                    if self.problem_description.unload:
                        sol.save(
                            self.path + f"sol_{i}.txt",
                            self.problem_description.test_psps_ids[i],
                        )
                    else:
                        sol.save(
                            self.path + f"sol_{i}.txt",
                            self.problem_description.test_psps[i].pb_id,
                        )
                else:
                    sol.save(self.path + f"sol_{i}.txt")

            print("Saving agent", self.path + "agent.pkl")
            agent.save(self.path + "agent.pkl")
            torch.save(alg.optimizer.state_dict(), self.path + "optimizer.pkl")
            self.save_state(self.path + "validator.pkl")

            self.criterion_ratio = cur_ratio
            print(f"Current ratio : {cur_ratio:.3f}")

            if self.compute_ortools:
                print(f"Saving the figure {self.path + 'best-ortools-ppo-cactus.png'}")
                self.current_scatter_fig.savefig(
                    self.path + "best-ortools-ppo-cactus.png"
                )

    def _get_ortools_criterion(self, i: int, ortools_strategy: str, criterion: str):
        save_path = self._ortools_solution_path(i, ortools_strategy)
        if self.psp:
            problem = self.validation_envs[i].problem
        else:
            problem = self.validation_envs[i].state

        previous_ortools_solution = self._ortools_read_solution(save_path, problem)
        if previous_ortools_solution is not None:
            return previous_ortools_solution

        if self.psp:
            criterion_value, schedule, optimal = get_ortools_criterion_psp(
                self.validation_envs[i],
                self.max_time_ortools,
                self.scaling_constant_ortools,
                ortools_strategy,
                criterion,
            )
        else:
            criterion_value, schedule, optimal = get_ortools_makespan_jssp(
                self.validation_envs[i].state.affectations,
                self.validation_envs[i].state.original_durations,
                self.env_specification.n_features,
                self.max_time_ortools,
                self.scaling_constant_ortools,
                ortools_strategy,
            )

        self._ortools_save_solution(
            save_path, problem, (criterion_value, schedule, optimal)
        )
        return criterion_value, schedule, optimal

    def _ortools_solution_path(self, i: int, ortools_strategy: str) -> str:
        return os.path.join(self.path, f"ortools_{ortools_strategy}_{i}.pkl")

    def _ortools_save_solution(self, file_path, problem, ortools_solution):
        with open(file_path, "wb") as f:
            pickle.dump((problem, ortools_solution), f)

    def _ortools_read_solution(self, file_path, current_problem):
        """Returns the solution if the file exists and if the instance is the same
        as the current problem. Returns None otherwise.
        """
        if not os.path.exists(file_path):
            return

        with open(file_path, "rb") as f:
            saved_problem, ortools_solution = pickle.load(f)

        if self.psp and current_problem == saved_problem:
            return ortools_solution

        if (
            not self.psp
            and np.all(current_problem.affectations == current_problem.affectations)
            and np.all(current_problem.durations == saved_problem.durations)
            and np.all(
                current_problem.original_durations == saved_problem.original_durations
            )
        ):
            return ortools_solution

    def _get_random_criterion(self, i):
        sol = self.random_agent.predict(self.validation_envs[i])
        if sol is None:
            return self.validation_envs[i].state.undoable_makespan
        return sol.get_criterion()

    # transform list of dicts to dict of lists
    def _list_to_dict(self, batch_list):
        batch_dict = {}
        for obs in batch_list:
            for key, value in obs.items():
                batch_dict.setdefault(key, []).append(value)
        return batch_dict

    def save_csv(self, name, criterion, optimal, schedule, sampled_jobs):
        f = open(self.path + name + ".csv", "w")
        writer = csv.writer(f)
        if sampled_jobs is not None:
            writer.writerow(["sampled jobs", sampled_jobs])
        writer.writerow(["criterion", criterion])
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
            header = ["all starts"]
            writer.writerow(header)
            for i in range(len(schedule[0])):
                writer.writerow([schedule[0][i]])
            writer.writerow([])
            header = ["all modes"]
            writer.writerow(header)
            for i in range(len(schedule[0])):
                writer.writerow([schedule[1][i]])
        f.close()

    def _evaluate_agent(self, agent):
        mean_criterion = 0
        if self.compute_ortools:
            ortools_mean_criterion = {
                ortools_strategy: 0 for ortools_strategy in self.ortools_strategies
            }
        random_mean_criterion = 0
        custom_mean_criterion = {agent.rule: 0 for agent in self.custom_agents}
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
                all_obs = agent.rebatch_obs(all_obs)
                all_actions = []
                for i in range(0, len(envs), self.batch_size):
                    bs = min(self.batch_size, len(envs) - i)
                    actions = agent.predict(
                        agent.get_obs(all_obs, list(range(i, i + bs))),
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

        solution = []
        for i in tqdm.tqdm(range(self.n_validation_env), desc="   evaluating         "):
            if self.batch_size == 0:
                obs, info = self.validation_envs[i].reset(soft=self.fixed_validation)
                # print("RESET")
                # print("graph", obs._graph)
                # print("prec edges", obs.edges("prec"))
                # print("rp edges", obs.edges("rp"))
                # print("ndata", obs.ndata())
                # print("global data", obs.global_data())
                done = False
                while not done:
                    action_masks = info["mask"].reshape(1, -1)
                    action_masks = decode_mask(action_masks)
                    # print("STEP ", i)
                    obs = agent.obs_as_tensor_add_batch_dim(obs)
                    # print("graph", obs._graph)
                    # print("prec edges", obs.edges("prec"))
                    # print("rp edges", obs.edges("rp"))
                    # print("ndata", obs.ndata())
                    # print("global data", obs.global_data())
                    action = agent.predict(
                        agent.preprocess(obs),
                        deterministic=True,
                        action_masks=action_masks,
                    )
                    obs, reward, done, _, info = self.validation_envs[i].step(
                        action.long().item()
                    )
            solution.append(self.validation_envs[i].get_solution())
            if solution[i] is not None:
                schedule = solution[i].schedule
                criterion = solution[i].get_criterion()

                if i == 0 and self.display_gantt:
                    self.gantt_rl_img = self.validation_envs[i].render_solution(
                        schedule
                    )

                if criterion < self.best_criterion_wheatley[i]:
                    self.best_criterion_wheatley[i] = criterion
                    self.save_csv(
                        f"wheatley_{i}",
                        criterion,
                        "unknown",
                        schedule,
                        self.validation_envs[i].sampled_jobs,
                    )

                self.last_ppo_criterions[i] = criterion

                mean_criterion += criterion / self.n_validation_env
            else:
                schedule = None
                state = self.validation_envs[i].state
                self.last_ppo_criterions[i] = state.undoable_criterion
                mean_criterion += state.undoable_criterion / self.n_validation_env
                if self.display_gantt:
                    self.gantt_rl_img = self.validation_envs[i].render_fail()

            if self.compute_ortools:
                for ortools_strategy in self.ortools_strategies:
                    if self.fixed_validation:
                        (
                            or_tools_criterion,
                            or_tools_schedule,
                            optimal,
                        ) = self.fixed_ortools[ortools_strategy][i]
                    else:
                        (
                            or_tools_criterion,
                            or_tools_schedule,
                            optimal,
                        ) = self._get_ortools_criterion(
                            i,
                            ortools_strategy,
                            self.problem_description.reward_model_config,
                        )

                    if ortools_strategy == self.averagistic_ortools_strategy:
                        self.last_ortools_criterions[i] = or_tools_criterion

                    if i == 0 and ortools_strategy == self.default_ortools_strategy:
                        if self.display_gantt:
                            self.gantt_or_img = self.validation_envs[i].render_solution(
                                or_tools_schedule, scaling=1.0
                            )
                        self.ortools_env_zero_is_optimal = optimal

                    if (
                        ortools_strategy == self.default_ortools_strategy
                        and or_tools_criterion < self.best_criterion_ortools[i]
                    ):
                        self.best_criterion_ortools[i] = or_tools_criterion
                        self.save_csv(
                            f"ortools_{i}",
                            or_tools_criterion,
                            optimal,
                            or_tools_schedule,
                            self.validation_envs[i].sampled_jobs,
                        )

                    ortools_mean_criterion[ortools_strategy] += (
                        or_tools_criterion / self.n_validation_env
                    )

            if self.fixed_random_validation:
                random_criterion = self.fixed_random[i]
            else:
                random_criterion = self._get_random_criterion(i)
            random_mean_criterion += random_criterion / self.n_validation_env

            for custom_agent in self.custom_agents:
                name = custom_agent.rule

                if self.fixed_validation:
                    criterion = self.fixed_custom_solutions[name][i]
                else:
                    solution = custom_agent.predict(
                        self.validation_envs[i].state.original_durations,
                        self.validation_envs[i].state.affectations,
                        self.problem_description.criterion,
                        self.validation_envs[i],
                    )
                    criterion = solution.get_criterion()
                    if custom_agent.rule == "MOPNR" and i == 0 and self.display_gantt:
                        self.gantt_mopnr_img = self.validation_envs[i].render_solution(
                            solution.schedule
                        )

                custom_mean_criterion[name] += criterion / self.n_validation_env

        print("--- mean_criterion=", mean_criterion, " ---")
        print("--- eval time=", time.time() - start_eval, "  ---")
        self.criterions.append(mean_criterion)
        if self.compute_ortools:
            for ortools_strategy in self.ortools_strategies:
                self.ortools_criterions[ortools_strategy].append(
                    ortools_mean_criterion[ortools_strategy]
                )
        self.random_criterions.append(random_mean_criterion)
        for custom_agent in self.custom_agents:
            self.custom_criterions[custom_agent.rule].append(
                custom_mean_criterion[custom_agent.rule]
            )
        return solution

    def _visdom_metrics(self, agent, alg):
        commandline = " ".join(sys.argv)
        html = f"""
            <div style="padding: 5px">
                <h4>Total actions: {alg.global_step}</h4>
                <code>{commandline}</code>
            </div>
        """
        self.vis.text(html, win="html", opts={"width": 372, "height": 336})

        X = list(range(len(self.criterions)))
        Y_list = [self.criterions, self.random_criterions]
        opts = {
            "title": "Validation criterion",
            "legend": ["Wheatley", "Random"],
            "linecolor": [[31, 119, 180], [255, 127, 14]],
        }
        if self.compute_ortools:
            Y2_list = [
                np.array(self.criterions)
                / np.array(self.ortools_criterions[self.default_ortools_strategy]),
                np.array(self.random_criterions)
                / np.array(self.ortools_criterions[self.default_ortools_strategy]),
            ]
            opts2 = {
                "legend": ["Wheatley / OR-tools", "Random / OR-tools"],
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
            for ortools_strategy_id, ortools_strategy in enumerate(
                self.ortools_strategies
            ):
                Y_list.append(self.ortools_criterions[ortools_strategy])
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
            Y = np.array(self.custom_criterions[name])
            Y_list.append(Y)
            if self.compute_ortools:
                Y2_list.append(
                    Y / np.array(self.ortools_criterions[self.default_ortools_strategy])
                )

            opts["linecolor"].append(agent_colors[custom_agent_id])
            opts["legend"].append(name)

            opts2["legend"].append(name + " / OR-tools")
            opts2["linecolor"].append(agent_colors[custom_agent_id])

        opts["linecolor"] = np.array(opts["linecolor"])
        self.vis.line(X=X, Y=np.array(Y_list).T, win="validation_criterion", opts=opts)
        # opts2["linecolor"] = np.array(opts2["linecolor"])
        # self.vis.line(X=X, Y=np.stack(Y2_list).T, win="validation_criterion_ratio", opts=opts2)

        if self.compute_ortools:
            # ratio to OR-tools
            opts = {
                "title": "Wheatley / OR-Tools",
                "legend": [],
            }
            Y_ratios = []
            for ortools_strategy in self.ortools_strategies:
                opts["legend"].append(f"Wheatley/OR-Tools {ortools_strategy}")
                opts["legend"].append(f"Min Wheatley/OR-Tools {ortools_strategy}")
                ratio_to_ortools = np.array(self.criterions) / np.array(
                    self.ortools_criterions[ortools_strategy]
                )
                Y_ratios.append(ratio_to_ortools)
                Y_ratios.append(np.minimum.accumulate(ratio_to_ortools))

            self.vis.line(
                X=X,
                Y=np.array(Y_ratios).T,
                win="ratio_to_ortools",
                opts=opts,
            )

            # distance to OR-tools
            opts = {"title": "Distance to OR-tools"}
            dist_to_ortools = np.array(self.criterions) - np.array(
                self.ortools_criterions[self.default_ortools_strategy]
            )
            self.vis.line(X=X, Y=dist_to_ortools, win="dist_to_ortools", opts=opts)

            # time to OR-tools
            wins = 0
            count = min(len(self.criterions), 100)
            for i in range(1, count + 1):
                ppo = self.criterions[-i]
                ortools = self.ortools_criterions[self.default_ortools_strategy][-i]
                if ppo <= ortools or np.isclose(ppo, ortools):
                    wins += 1
            pct = 100 * wins / count
            self.time_to_ortools.append(pct)
            opts = {"title": "Time to OR-tools %"}
            self.vis.line(
                X=X, Y=np.array(self.time_to_ortools), win="time_to_ortools", opts=opts
            )

            # Plot OR-Tools solutions vs Wheatley solutions.
            # Visdom is a little bit limited so we build the plot ourself using matplotlib
            # and save the image on disk. We can directly plot this image with visdom.
            # This is a bit hacky but it works.
            fig, ax = plt.subplots(figsize=(6, 6))
            ppo_criterions = np.array(self.last_ppo_criterions)
            ortools_criterions = np.array(self.last_ortools_criterions)
            ax.scatter(
                ppo_criterions,
                ortools_criterions,
                c=-ppo_criterions / ortools_criterions,
                cmap="viridis",
            )
            ax.plot(
                [0, max(self.last_ppo_criterions + self.last_ortools_criterions)],
                [0, max(self.last_ppo_criterions + self.last_ortools_criterions)],
                color="red",
            )
            ax.set_xlabel("Wheatley criterion")
            ax.set_ylabel("OR-Tools criterion")
            ax.set_title(
                f"OR-Tools {self.averagistic_ortools_strategy} vs Wheatley - {(ppo_criterions / ortools_criterions).mean():.2f}"
            )
            fig.savefig(self.path + "ortools-ppo-cactus.png")

            if self.current_scatter_fig is not None:
                plt.close(self.current_scatter_fig)
            self.current_scatter_fig = fig

            # Load the image into a numpy array and pass it to visdom.
            image = np.array(Image.open(self.path + "ortools-ppo-cactus.png"))
            image = image.transpose(2, 0, 1)
            image = image[:3, :, :]  # Remove alpha channel.
            self.vis.image(
                image,
                win="ortools-ppo-cactus",
                opts={"caption": "OR-Tools vs Wheatley"},
            )

        if self.first_callback:
            self.first_callback = False
            return

        if self.compute_ortools:
            if self.criterion_ratio <= (ppo_criterions / ortools_criterions).mean():
                image = np.array(Image.open(self.path + "best-ortools-ppo-cactus.png"))
            else:
                image = np.array(Image.open(self.path + "ortools-ppo-cactus.png"))
            image = image.transpose(2, 0, 1)
            image = image[:3, :, :]  # Remove alpha channel.
            self.vis.image(
                image,
                win="best-ortools-ppo-cactus",
                opts={"caption": "Best OR-Tools vs Wheatley"},
            )

        self.entropy_losses.append(
            # alg.ent_coef * alg.logger.name_to_value["train/entropy_loss"]
            alg.logger.name_to_value["train/entropy_loss"]
        )
        self.policy_gradient_losses.append(
            alg.logger.name_to_value["train/policy_gradient_loss"]
        )
        self.value_losses.append(
            # alg.vf_coef * alg.logger.name_to_value["train/value_loss"]
            alg.logger.name_to_value["train/value_loss"]
        )
        self.losses.append(alg.logger.name_to_value["train/loss"])
        self.approx_kls.append(alg.logger.name_to_value["train/approx_kl"])
        self.clip_fractions.append(alg.logger.name_to_value["train/clip_fraction"])
        self.explained_variances.append(
            alg.logger.name_to_value["train/explained_variance"]
        )
        self.return_variances.append(alg.logger.name_to_value["train/return_variance"])
        self.value_variances.append(alg.logger.name_to_value["train/value_variance"])

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
        if self.debug_net:
            for k in alg.logger.name_to_value.keys():
                if "net/var_" in k:
                    if k not in self.variances:
                        self.variances[k] = []
                    self.variances[k].append(alg.logger.name_to_value[k])

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

        # variances
        if self.debug_net:
            X = list(range(1, len(next(iter(self.variances.values()))) + 1))
            Y_list = list(self.variances.values())
            opts = {"legend": list(self.variances.keys()), "title": "net variances"}
            self.vis.line(X=X, Y=np.array(Y_list).T, win="variances", opts=opts)

        X = list(range(1, len(self.return_variances) + 1))
        Y_list = [self.return_variances, self.value_variances]
        opts = {
            "legend": ["return_variances", "value_variances"],
            "title": "critic variances",
        }
        self.vis.line(X=X, Y=np.array(Y_list).T, win="critic_variances", opts=opts)

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
        """realistic > averagistic > others"""
        if "realistic" in self.ortools_strategies:
            return "realistic"
        elif "averagistic" in self.ortools_strategies:
            return "averagistic"
        else:
            return self.ortools_strategies[0]

    @property
    def averagistic_ortools_strategy(self) -> str:
        """averagistic > realistic > others"""
        if "averagistic" in self.ortools_strategies:
            return "averagistic"
        elif "realistic" in self.ortools_strategies:
            return "realistic"
        else:
            return self.ortools_strategies[0]

    def save_state(self, filepath: str):
        if self.compute_ortools:
            validator_state = {
                "criterion_ratio": self.criterion_ratio,
                "criterions": self.criterions,
                "ortools_criterions": self.ortools_criterions,
                "random_criterions": self.random_criterions,
                "custom_criterions": self.custom_criterions,
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
                "all_or_tools_criterion": self.all_or_tools_criterion,
                "all_or_tools_schedule": self.all_or_tools_schedule,
                "time_to_ortools": self.time_to_ortools,
                "best_criterion_wheatley": self.best_criterion_wheatley,
                "best_criterion_ortools": self.best_criterion_ortools,
                "ortools_env_zero_is_optimal": self.ortools_env_zero_is_optimal,
            }
        else:
            validator_state = {
                "criterion_ratio": self.criterion_ratio,
                "criterions": self.criterions,
                "random_criterions": self.random_criterions,
                "custom_criterions": self.custom_criterions,
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
                "all_or_tools_criterion": self.all_or_tools_criterion,
                "all_or_tools_schedule": self.all_or_tools_schedule,
                "best_criterion_wheatley": self.best_criterion_wheatley,
            }
        with open(filepath, "wb") as f:
            pickle.dump(validator_state, f)

    def reload_state(self, filepath: str) -> "AgentValidator":
        with open(filepath, "rb") as f:
            validator_state = pickle.load(f)

        self.criterion_ratio = validator_state["criterion_ratio"]
        self.criterions = validator_state["criterions"]
        if self.compute_ortools:
            self.ortools_criterions = validator_state["ortools_criterions"]
        self.random_criterions = validator_state["random_criterions"]
        self.custom_criterions = validator_state["custom_criterions"]
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
        self.all_or_tools_criterion = validator_state["all_or_tools_criterion"]
        self.all_or_tools_schedule = validator_state["all_or_tools_schedule"]
        if self.compute_ortools:
            self.time_to_ortools = validator_state["time_to_ortools"]
        self.best_criterion_wheatley = validator_state["best_criterion_wheatley"]
        if self.compute_ortools:
            self.best_criterion_ortools = validator_state["best_criterion_ortools"]
            self.ortools_env_zero_is_optimal = validator_state[
                "ortools_env_zero_is_optimal"
            ]

        return self
