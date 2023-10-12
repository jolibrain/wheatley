#
# Wheatley
# Copyright (c) 2023 Jolibrain
# Authors:
#    Guillaume Infantes <guillaume.infantes@jolibrain.com>
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
import pathlib
import glob


def load_problem(
    problem_file,
    taillard_offset=False,
    deterministic=True,
    load_from_job=0,
    load_max_jobs=-1,
    generate_bounds=-1.0,
):
    # Customized problem loader
    # - support for bounded duration uncertainty
    # - support for unattributed machines
    # - support for columns < number of machines

    print("generate_bounds=", generate_bounds)

    if not deterministic:
        print("Loading problem with uncertainties, using customized format")
        if generate_bounds > 0:
            print("Generating random duration bounds of ", generate_bounds, " %")

    with open(problem_file, "r") as f:
        line = next(f)
        while line[0] == "#":
            line = next(f)

        # header
        header = line
        head_list = [int(i) for i in header.split()]
        assert len(head_list) == 2
        n_j = head_list[0]
        n_m = head_list[1]

        line = next(f)
        while line[0] == "#":
            line = next(f)

        # matrix of durations
        np_lines = []
        for j in range(n_j):
            dur_list = []
            for i in line.split():
                add_dur = float(i)
                if add_dur == 0:
                    add_dur = 0.1
                elif add_dur < 0:
                    add_dur = -1.0
                dur_list.append(add_dur)
            while len(dur_list) < n_m:
                dur_list.append(-1.0)
            np_lines.append(np.array(dur_list))
            line = next(f)
        durations = np.stack(np_lines)

        if deterministic:
            durations = np.expand_dims(durations, axis=2)
            durations = np.repeat(durations, 4, axis=2)
        elif generate_bounds > 0.0:
            mode_durations = durations
            min_durations = np.subtract(
                durations,
                generate_bounds * durations,
                out=durations.copy(),
                where=durations != -1,
            )
            max_durations = np.add(
                durations,
                generate_bounds * durations,
                out=durations.copy(),
                where=durations != -1,
            )
            real_durations = np.zeros((n_j, n_m)) - 1
            durations = np.stack(
                [real_durations, min_durations, max_durations, mode_durations], axis=2
            )
            # sys.exit()
        else:
            mode_durations = durations

            while line[0] == "#":
                line = next(f)

            np_lines = []
            for j in range(n_j):
                dur_list = []
                for i in line.split():
                    add_dur = float(i)
                    if add_dur == 0:
                        add_dur = 0.1
                    elif add_dur < 0:
                        add_dur = -1.0
                    dur_list.append(add_dur)
                while len(dur_list) < n_m:
                    dur_list.append(-1.0)
                np_lines.append(np.array(dur_list))
                line = next(f)
            min_durations = np.stack(np_lines)

            while line[0] == "#":
                line = next(f)

            np_lines = []
            for j in range(n_j):
                dur_list = []
                for i in line.split():
                    add_dur = float(i)
                    if add_dur == 0:
                        add_dur = 0.1
                    elif add_dur < 0:
                        add_dur = -1.0
                    dur_list.append(add_dur)
                while len(dur_list) < n_m:
                    dur_list.append(-1.0)
                np_lines.append(np.array(dur_list))
                line = next(f)
            max_durations = np.stack(np_lines)

            real_durations = np.zeros((n_j, n_m)) - 1

            durations = np.stack(
                [real_durations, min_durations, max_durations, mode_durations], axis=2
            )

        while line[0] == "#":
            line = next(f)

        # matrix of affectations
        if taillard_offset:
            toffset = 1
        else:
            toffset = 0
        np_lines = []
        for j in range(n_j):
            aff_list = [
                int(i) - toffset for i in line.split()
            ]  # Taillard spec has machines id start at 1
            while len(aff_list) < n_m:
                aff_list.append(-1)
            np_lines.append(np.array(aff_list))
            line = next(f, "")
            if line == "":
                break
        affectations = np.stack(np_lines)

        if load_max_jobs == -1:
            load_max_jobs = n_j

        affectations = affectations[load_from_job : load_from_job + load_max_jobs]
        durations = durations[load_from_job : load_from_job + load_max_jobs]

        check_sanity(affectations, durations)

        return len(affectations), n_m, affectations, durations


def load_taillard_problem(problem_file, taillard_offset=True, deterministic=True):
    # http://jobshop.jjvh.nl/explanation.php#taillard_def

    if not deterministic:
        print("Loading problem with uncertainties, using extended taillard format")

    with open(problem_file, "r") as f:
        line = next(f)
        while line[0] == "#":
            line = next(f)

        # header
        header = line
        head_list = [int(i) for i in header.split()]
        assert len(head_list) == 2
        n_j = head_list[0]
        n_m = head_list[1]

        line = next(f)
        while line[0] == "#":
            line = next(f)

        # matrix of durations
        np_lines = []
        for j in range(n_j):
            dur_list = [float(i) for i in line.split()]
            np_lines.append(np.array(dur_list))
            line = next(f)
        durations = np.stack(np_lines)

        if deterministic:
            durations = np.expand_dims(durations, axis=2)
            durations = np.repeat(durations, 4, axis=2)
        else:
            mode_durations = durations

            while line[0] == "#":
                line = next(f)

            np_lines = []
            for j in range(n_j):
                dur_list = [float(i) for i in line.split()]
                np_lines.append(np.array(dur_list))
                line = next(f)
            min_durations = np.stack(np_lines)

            while line[0] == "#":
                line = next(f)

            np_lines = []
            for j in range(n_j):
                dur_list = [float(i) for i in line.split()]
                np_lines.append(np.array(dur_list))
                line = next(f)
            max_durations = np.stack(np_lines)

            real_durations = np.zeros((n_j, n_m)) - 1

            durations = np.stack(
                [real_durations, min_durations, max_durations, mode_durations], axis=2
            )

        while line[0] == "#":
            line = next(f)

        # matrix of affectations
        if taillard_offset:
            toffset = 1
        else:
            toffset = 0
        np_lines = []
        for j in range(n_j):
            aff_list = [
                int(i) - toffset for i in line.split()
            ]  # Taillard spec has machines id start at 1
            np_lines.append(np.array(aff_list))
            line = next(f, "")
            if line == "":
                break
        affectations = np.stack(np_lines)

        return n_j, n_m, affectations, durations
