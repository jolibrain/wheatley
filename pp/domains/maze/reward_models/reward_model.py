import torch
import math


class ShortestPathRewardModel:
    def __init__(self, env_specification, goal_reward=0.0, nonchrono=None):
        self.reward_shaping = nonchrono is not None and nonchrono == "path"
        # self.reward_shaping = False
        self.nonchrono = nonchrono
        self.env_specification = env_specification
        self.unit_cost = 1.0 / (env_specification.max_n_nodes * 1)
        self.move_cost = 1.0 / env_specification.max_n_nodes
        self.step_cost = 1.0 / env_specification.max_n_steps
        self.reward_shaping_rew = 1.0 / math.sqrt(env_specification.max_n_nodes)
        self.goal_reward = float(goal_reward)
        if self.reward_shaping:
            self.fail_cost = 2.0
        else:
            self.fail_cost = (
                2.0  # worst path cost is 1, and we add 1-n_neigh_normalized
            )

        # non chrono2 should get:
        # some reward for having neighbors adjacent in list
        # some cost for steps
        # some extra cost for failure
        # some some cost for final path length

    def evaluate(self, state, current_reward, data_from_previous_state):
        if getattr(state, "just_reached_goal", False):
            state.just_reached_goal = False
            state.step_cost_multiplier = 1.0
            return self.goal_reward
        if state.succeeded():
            # not_in_path = torch.ones_like(state.selected, dtype=bool)
            # not_in_path[state.path] = False
            # print(
            #     f"succeded\n start = {state.selected[state.start]}\n goal = {state.selected[state.goal]}\n path = {state.selected[state.path]}\n nopath = {state.selected[not_in_path]}\n nsteps = {state.n_steps}\n sol : {state.env.problem.solvedMaze.solution} {state.env.problem.solvedMaze.solution.shape[0]}"
            # )
            state.step_cost_multiplier = 1.0
            if self.nonchrono in ["path", "wpr"]:
                if self.reward_shaping:
                    return (
                        -len(state.path) * self.move_cost
                        - data_from_previous_state * self.reward_shaping_rew
                    )
                else:
                    return -len(state.path) * self.move_cost
            return 0.0
        if state.failed():
            state.step_cost_multiplier = 1.0
            if self.nonchrono in ["path"]:
                if self.reward_shaping:
                    return (
                        -self.fail_cost
                        - data_from_previous_state * self.reward_shaping_rew
                    )
                else:
                    return -self.fail_cost + state.n_neigh * self.move_cost
            return -self.fail_cost  # - current_reward
        cost = -self.step_cost * getattr(state, "step_cost_multiplier", 1.0)
        state.step_cost_multiplier = 1.0
        if self.reward_shaping:
            return (
                cost
                + (state.n_neigh - data_from_previous_state) * self.reward_shaping_rew
            )
        else:
            return cost

    def get_data_for_reward(self, state):
        return state.n_neigh
