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

import numpy as np

from utils.utils import (
    generate_deterministic_problem,
    generate_problem_distrib,
    generate_problem_durations,
)


class JSSPDescription:
    def __init__(
        self,
        transition_model_config,
        reward_model_config,
        deterministic,
        fixed,
        affectations=None,
        durations=None,
        n_jobs=None,
        n_machines=None,
        max_duration=None,
        duration_mode_bounds=None,
        duration_delta=None,
    ):
        """
        There are 4 types of ProblemDescription:
         - Fixed problem, deterministic: specify affectations and durations (where durations is of shape
           [n_jobs, n_machines] or [n_jobs, n_machines, 4] where durations[:, :, 0] == ... == durations[:, :, 3])
         - Fixed problem, stochastic: specify affectations and durations (where durations is of shape
           [n_jobs, n_machines, 4] and durations[:, :, 1] != durations[:, :, 2])
         - Any problem, deterministic: specify n_jobs, n_machines, max_duration
         - Any problem, stochastic: Specify n_jobs, n_machines, duration_mode_bound, duration_delta
        """

        # Checking the consistency of the inputs
        self.check_consistency(affectations, durations, n_jobs, n_machines)
        if n_jobs is not None and n_machines is not None:
            self.n_jobs, self.n_machines = n_jobs, n_machines
        else:
            self.n_jobs, self.n_machines = affectations.shape[0], affectations.shape[1]

        # First case: fixed and deterministic
        if fixed and deterministic:
            if affectations is not None and durations is not None:
                if (
                    (durations[:, :, 0] != durations[:, :, 1]).any()
                    or (durations[:, :, 1] != durations[:, :, 2]).any()
                    or (durations[:, :, 2] != durations[:, :, 3]).any()
                ):
                    raise Exception(
                        "Problem is specified to be deterministic, but you provided stochastic durations"
                    )
                self.affectations = affectations
                self.durations = durations
            else:
                if max_duration is None:
                    raise Exception(
                        "Please provide max_duration to generate a deterministic problem"
                    )
                self.affectations, self.durations = generate_deterministic_problem(
                    self.n_jobs, self.n_machines, max_duration
                )
            self.deterministic = True
            self.fixed = True
            self.name = f"{self.n_jobs}j{self.n_machines}m_fixed_deterministic"
            self.max_duration = None
            self.duration_mode_bounds = None
            self.duration_delta = None

        # Second case: fixed and stochastic
        elif fixed and not deterministic:
            if affectations is not None and durations is not None:
                if (
                    (durations[:, :, 0] == durations[:, :, 1]).all()
                    and (durations[:, :, 1] == durations[:, :, 2]).all()
                    and (durations[:, :, 2] == durations[:, :, 3]).all()
                ):
                    raise Exception(
                        "Problem is specified to be stochastic, but you provided deterministic durations"
                    )
                self.affectations = affectations
                self.durations = durations
            else:
                if duration_mode_bounds is None or duration_delta is None:
                    raise Exception(
                        "Please provide duration_mode_bounds and duration_delta to generate a stochastic problem"
                    )
                self.affectations, self.durations = generate_problem_distrib(
                    self.n_jobs, self.n_machines, duration_mode_bounds, duration_delta
                )
            # Generate a first version of the durations, to have a totally instantiated duration
            self.durations = generate_problem_durations(self.durations)
            # But we still want to regenerate durations at each sampling. This can be deactivated for totally fixed problem
            self.regenerate_durations = True

            self.deterministic = False
            self.fixed = True
            self.name = f"{self.n_jobs}j{self.n_machines}m_fixed_stochastic"
            self.max_duration = None
            self.duration_mode_bounds = None
            self.duration_delta = None

        # Third case
        elif not fixed and deterministic:
            if durations is not None or affectations is not None:
                raise Exception(
                    "Please don't provide affectations or durations for unfixed problem"
                )
            if max_duration is None:
                raise Exception(
                    "Please provide a valid max_duration for unfixed deterministic problem"
                )
            self.max_duration = max_duration
            self.deterministic = True
            self.fixed = False
            self.name = f"{self.n_jobs}j{self.n_machines}m_any_deterministic"
            self.affectations = None
            self.durations = None
            self.duration_mode_bounds = None
            self.duration_delta = None

        # Fourth case
        elif not fixed and not deterministic:
            if durations is not None or affectations is not None:
                raise Exception(
                    "Please don't provide affectations or durations for unfixed problem"
                )
            if duration_mode_bounds is None or duration_delta is None:
                raise Exception(
                    "Please provide a valid duration_mode_bounds or duration_delta for unfixed stochastic problem"
                )
            self.duration_mode_bounds = duration_mode_bounds
            self.duration_delta = duration_delta
            self.deterministic = False
            self.fixed = False
            self.name = f"{self.n_jobs}j{self.n_machines}m_any_stochastic"
            self.durations = None
            self.affectations = None
            self.max_duration = None

        self.transition_model_config = transition_model_config
        self.reward_model_config = reward_model_config

    def check_consistency(self, affectations, durations, n_jobs, n_machines):
        # Consistency of affectations and durations
        if (affectations is not None and durations is None) or (
            durations is not None and affectations is None
        ):
            raise Exception(
                "affectations xor durations is None. Please provide affectations and durations or none of them"
            )

        # Consistency of n_jobs and n_machines in regard to specified affectations and durations
        if affectations is not None and durations is not None:
            cur_n_j, cur_n_m = affectations.shape[0], affectations.shape[1]
            if cur_n_j != durations.shape[0] or cur_n_m != durations.shape[1]:
                raise Exception(
                    "Please provide affectations and durations of matching shapes"
                )
            if (n_jobs is not None and n_jobs != cur_n_j) or (
                n_machines is not None and n_machines != cur_n_m
            ):
                raise Exception(
                    "Please provide n_jobs ("
                    + str(n_jobs)
                    + " vs "
                    + str(cur_n_j)
                    + ") and n_machines ("
                    + str(n_machines)
                    + " vs "
                    + str(cur_n_m)
                    + ") that are consistent with affectations and durations"
                )
        elif n_jobs is None or n_machines is None:
            raise Exception(
                "Please provide n_jobs and n_machines or affectations and durations"
            )

    def sample_problem(self):
        """Returns an instance of a problem corresponding to the problem description"""
        if self.fixed:
            if self.deterministic:
                return self.affectations, self.durations
            else:
                if self.regenerate_durations:
                    return self.affectations, generate_problem_durations(self.durations)
                else:
                    return self.affectations, self.durations
        else:
            if self.deterministic:
                return generate_deterministic_problem(
                    self.n_jobs, self.n_machines, self.max_duration
                )
            else:
                affectations, durations = generate_problem_distrib(
                    self.n_jobs,
                    self.n_machines,
                    self.duration_mode_bounds,
                    self.duration_delta,
                )
                return affectations, generate_problem_durations(durations)

    def get_frozen_version(self):
        """
        This method is used when a problem description should be fixed.
        This is used on test purpose within the test script or inside validation callbacks
        """
        affectations, durations = self.sample_problem()
        problem_description = ProblemDescription(
            transition_model_config=self.transition_model_config,
            reward_model_config=self.reward_model_config,
            deterministic=self.deterministic,
            fixed=True,
            affectations=affectations,
            durations=durations,
        )
        # We want a totally frozen problem, not a "fixed" stochastic version.
        problem_description.regenerate_durations = False
        return problem_description

    def print_self(self):
        print(
            f"==========Problem Description ==========\n"
            f"Problem size:                     {self.n_jobs} x {self.n_machines}\n"
            f"Transition model:                 {self.transition_model_config}\n"
            f"Reward model:                     {self.reward_model_config}\n"
            f"Fixed/Unfixed:                    {'Fixed' if self.fixed else 'Unfixed'}\n"
            f"Deterministic/Stochastic:         {'Deterministic' if self.deterministic else 'Stochastic'}\n"
        )
        if self.fixed:
            print(
                "ntasks: ",
                (np.nonzero(np.where(self.affectations == -1, 0, 1)))[0].shape[0],
            )
            print("Affectations")
            print(self.affectations)
            print("Durations")
            if self.deterministic:
                print(self.durations[:, :, 0])
            else:
                print("Min")
                print(self.durations[:, :, 1])
                print("Mode")
                print(self.durations[:, :, 3])
                print("Max")
                print(self.durations[:, :, 2])
