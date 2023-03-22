#
# Wheatley
# Copyright (c) 2023 Jolibrain
# Authors:
#    Antoine Jacquet <antoine.jacquet@jolibrain.com>
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

import numpy as np
import visdom
from models.random_agent import RandomAgent
from utils.utils_testing import get_ortools_makespan
from utils.utils import job_and_task_to_node, node_to_job_and_task
import sys

# hide Visdom deprecation warnings
import warnings

warnings.simplefilter("ignore", DeprecationWarning)


class MuZeroCallback:
    def __init__(
        self,
        problem_description,
        env_specification,
        training_specification,
    ):
        # Parameters
        self.problem_description = problem_description
        self.env_specification = env_specification
        self.training_specification = training_specification

        # Aliases
        self.ortools_strategy = training_specification.ortools_strategy
        self.max_n_jobs = env_specification.max_n_jobs
        self.max_n_machines = env_specification.max_n_machines
        self.max_time_ortools = training_specification.max_time_ortools
        self.scaling_constant_ortools = training_specification.scaling_constant_ortools

        # Comparative agents
        self.random_agent = RandomAgent(self.max_n_jobs, self.max_n_machines)

        # Inner variables
        self.makespans = []
        self.ortools_makespans = []
        self.random_makespans = []
        self.time_to_ortools = []
        self.charts = {}
        self.commandline = " ".join(sys.argv)

    def __getstate__(self):
        state = self.__dict__.copy()
        # prevent MuZero from serializing Visdom during checkpoint
        state.pop("vis", None)
        return state

    def get_ortools_schedule(self, env):
        _, ortools_schedule = get_ortools_makespan(
            env.state.affectations,
            env.state.original_durations,
            self.max_time_ortools,
            self.scaling_constant_ortools,
            self.ortools_strategy,
        )
        return ortools_schedule

    def get_ortools_actions(self, env, ortools_schedule):
        nodes_time = ortools_schedule.flatten()
        nodes_machine = env.state.affectations.flatten()
        nodes_affected = env.state.affected.flatten()

        # for each machine, get the first node time to be scheduled
        machine_next = {}
        for node_id, machine_id in enumerate(nodes_machine):
            if machine_id == -1:
                continue
            if nodes_affected[node_id]:
                continue
            previous = machine_next.get(machine_id, float("inf"))
            current = nodes_time[node_id]
            if current < previous:
                machine_next[machine_id] = current

        # for each valid action, check if it is the minimum node time for this machine
        mask = env.action_masks()
        valid_actions = [node for node, masked in enumerate(mask) if masked == True]
        ortools_actions = []
        for action in valid_actions:
            machine_id = nodes_machine[action]
            if nodes_time[action] != machine_next[machine_id]:
                continue
            ortools_actions.append(action)

        return ortools_actions

    def get_ortools_trajectory(self, env):
        schedule = self.get_ortools_schedule(env)
        trajectory = []
        env.reset(soft=True)
        while not env.done():
            actions = self.get_ortools_actions(env, schedule)
            action = np.random.choice(actions)
            env.step(action)
            trajectory.append(action)
        env.reset(
            soft=True
        )  # we want MuZero to apply the trajectory to the same problem
        return trajectory

    def visdom(self, envs, metrics, config):
        if metrics["training_step"] == 0:
            return

        # first connection to Visdom
        if not hasattr(self, "vis"):
            self.vis = visdom.Visdom(env=self.training_specification.display_env)

        # html
        types = ["int", "bool", "float", "list", "NoneType"]
        table = ""
        for k, v in config.__dict__.items():
            if not type(v).__name__ in types or k == "action_space":
                continue
            table += "<tr><td>" + k + "</td><td>" + str(v) + "</td></tr>"
        html = f"""
            <div style="padding: 5px">
                <code>{self.commandline}</code>
                <hr />
                <table width="100%">{table}</table>
            </div>
        """
        self.vis.text(html, win="html", opts={"width": 372, "height": 336})

        # mean makespans
        mean_makespan = 0
        ortools_mean_makespan = 0
        random_mean_makespan = 0

        # for each validation env
        for i, env in enumerate(envs):

            # env contains the solution found by MuZero
            solution = env.get_solution()
            schedule = solution.schedule
            makespan = solution.get_makespan()
            mean_makespan += makespan / len(envs)

            # also solve it with OR-tools
            ortools_makespan, ortools_schedule = get_ortools_makespan(
                env.state.affectations,
                env.state.original_durations,
                self.max_time_ortools,
                self.scaling_constant_ortools,
                self.ortools_strategy,
            )
            ortools_mean_makespan += ortools_makespan / len(envs)

            # also solve it with random agent
            random_solution = self.random_agent.predict(env)
            random_makespan = random_solution.get_makespan()
            random_mean_makespan += random_makespan / len(envs)

            # only draw first gantts
            if i == 0:
                gantt_rl_img = env.render_solution(schedule)
                gantt_or_img = env.render_solution(ortools_schedule, scaling=1.0)

        # makespans
        self.makespans.append(mean_makespan)
        self.ortools_makespans.append(ortools_mean_makespan)
        self.random_makespans.append(random_mean_makespan)
        X = list(range(len(self.makespans)))
        Y_list = [self.makespans, self.random_makespans, self.ortools_makespans]
        opts = {
            "legend": ["MuZero", "Random", "OR-tools"],
            "linecolor": np.array([[31, 119, 180], [255, 127, 14], [44, 160, 44]]),
        }
        self.vis.line(X=X, Y=np.array(Y_list).T, win="validation_makespan", opts=opts)

        # ratio to OR-tools
        opts = {"title": "MuZero / OR-tools"}
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
            if self.makespans[-i] <= self.ortools_makespans[-i]:
                wins += 1
        pct = 100 * wins / count
        self.time_to_ortools.append(pct)
        opts = {"title": "Time to OR-tools %"}
        self.vis.line(
            X=X, Y=np.array(self.time_to_ortools), win="time_to_ortools", opts=opts
        )

        # draw other metrics
        for key, value in metrics.items():

            # first time we see it
            if not key in self.charts:
                self.charts[key] = []

            # append last metric
            Y = self.charts[key]
            Y.append(value)

            # update chart
            self.vis.line(X=X, Y=Y, win=key, opts={"title": key})

        # gantts
        self.vis.image(
            gantt_rl_img, opts={"caption": "Gantt RL schedule"}, win="rl_schedule"
        )
        self.vis.image(
            gantt_or_img, opts={"caption": "Gantt OR-Tools schedule"}, win="or_schedule"
        )
