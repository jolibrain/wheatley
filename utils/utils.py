from copy import deepcopy

import numpy as np
import torch


def find_last_in_batch(start_index, bi, batch_indices):
    index = start_index
    while index < batch_indices.shape[0] - 1 and int(batch_indices[index + 1].item()) == bi:
        index += 1
    return index


def get_exp_name(args):
    exp_name = (
        f"{args.n_j}j{args.n_m}m_{args.duration_type}_{args.seed}seed_{args.transition_model_config}_"
        + f"{args.reward_model_config}_{args.gconv_type}_{args.graph_pooling}"
    )
    if args.dont_normalize_input:
        exp_name += "_DNI"
    if args.fixed_problem:
        exp_name += "_FP"
    if args.load_problem:
        exp_name += "_" + args.load_problem.replace(".txt", "").replace("/", "_")
    if args.freeze_graph:
        exp_name += "_FG"
    if args.insertion_mode == "no_forced_insertion":
        exp_name += "_NFI"
    if args.insertion_mode == "full_forced_insertion":
        exp_name += "_FFI"
    if args.insertion_mode == "choose_forced_insertion":
        exp_name += "_CFI"
    if args.insertion_mode == "slot_locking":
        exp_name += "_SL"
    if args.exp_name_appendix is not None:
        exp_name += "_" + args.exp_name_appendix
    return exp_name


def get_n_features(input_list, max_n_jobs, max_n_machines):
    if "one_hot_machine_id" in input_list:
        input_list.remove("one_hot_machine_id")
    n_features = 4 * (2 + len(input_list))
    if "one_hot_job_id" in input_list:
        n_features += max_n_jobs - 4
    n_features += max_n_machines

    return n_features


def get_path(arg_path, exp_name):
    path = "saved_networks/" + exp_name if arg_path == "saved_networks/default_net" else arg_path
    return path


def generate_deterministic_problem(n_jobs, n_machines, high):
    """
    Generate a random intance of a JSS problem, of size (n_jobs, n_machines), with times comprised between specified
    lower and higher bound.
    Note that durations is of shape[n_jobs, n_machines, 4], but its only 4 repetitions of an array of shape
    [n_jobs, n_machines]
    """
    durations = np.random.randint(low=1, high=high, size=(n_jobs, n_machines, 1))
    durations = np.repeat(durations, repeats=4, axis=2)
    affectations = np.expand_dims(np.arange(0, n_machines), axis=0)
    affectations = affectations.repeat(repeats=n_jobs, axis=0)
    affectations = _permute_rows(affectations)
    return affectations, durations


def generate_problem_distrib(n_jobs, n_machines, duration_mode_bounds, duration_delta):
    """
    Generate a problem distribution, using duration_mode_bounds = [v1, v2] and duration_delta=[d1, d2].

    The durations variable is a numpy.array of shape [n_jobs, n_machines, 4]
    durations[:, :, 0] will store real durations (and is left unaffected).
    (durations[:, :, 1], durations[:, :, 3], durations[:, :, 2]) will be the (left, mode, right) value of a triangular
    distribution (see https://numpy.org/doc/stable/reference/random/generated/numpy.random.triangular.html)

    durations[:, :, 3] is randomly sampled in the [v1, v2] interval
    durations[:, :, 1] is randomly sampled in the [durations[:, :, 3] - d1, durations[:, :, 3] - 1] interval
    durations[:, :, 2] is randomly sampled in the [durations[:, :, 3] + 1, durations[:, :, 3] + d2] interval
    """
    durations = np.empty((n_jobs, n_machines, 4), dtype=np.int32)

    # Sampling the modes, using duration_mode_bounds
    durations[:, :, 3] = np.random.randint(duration_mode_bounds[0], duration_mode_bounds[1], size=(n_jobs, n_machines))
    for j in range(n_jobs):
        for m in range(n_machines):
            # Sampling the left
            dd = np.random.randint(1, duration_delta[0])
            durations[j, m][1] = max(1, durations[j, m][3] - dd)
            # Sampling the right
            dd = np.random.randint(1, duration_delta[1])
            durations[j, m][2] = durations[j, m][3] + dd

    # Each row of the affectations array is a permutation of [0, n_machines[
    affectations = np.expand_dims(np.arange(0, n_machines), axis=0)
    affectations = affectations.repeat(repeats=n_jobs, axis=0)
    affectations = _permute_rows(affectations)
    return affectations, durations


def generate_problem_durations(durations):
    """
    Generate the problem real durations, using a triangular distribution.
    durations[:, :, 0] with a triangular distribution using (durations[:, :, 1], durations[:, :, 3], durations[:, :, 2])
    as (left, mode, right) tuple (see https://numpy.org/doc/stable/reference/random/generated/numpy.random.triangular.html)
    """
    ret_durations = deepcopy(durations)
    n_jobs = durations.shape[0]
    n_machines = durations.shape[1]
    for j in range(n_jobs):
        for m in range(n_machines):
            if durations[j, m, 1] == durations[j, m, 2]:
                ret_durations[j, m, 0] = durations[j, m, 1]
            else:
                if durations[j, m][1] > durations[j, m][3]:
                    print(durations[j, m][1], durations[j, m][3])
                ret_durations[j, m, 0] = np.random.triangular(durations[j, m][1], durations[j, m][3], durations[j, m][2])
    return ret_durations


def _permute_rows(x):
    """
    x is a bidimensional numpy array
    """
    ix_i = np.tile(np.arange(x.shape[0]), (x.shape[1], 1)).transpose()
    ix_j = np.random.sample(x.shape).argsort(axis=1)
    return x[ix_i, ix_j]


def generate_data(n_j, n_m, max_duration, seed=200, n_problems=100):
    np.random.seed(seed)
    data = np.array([generate_deterministic_problem(n_j, n_m, max_duration) for _ in range(n_problems)])
    return data


def node_to_job_and_task(node_id, n_machines):
    return node_id // n_machines, node_id % n_machines


def job_and_task_to_node(job_id, task_id, n_machines):
    return job_id * n_machines + task_id


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

            durations = np.stack([real_durations, min_durations, max_durations, mode_durations], axis=2)

        while line[0] == "#":
            line = next(f)

        # matrix of affectations
        if taillard_offset:
            toffset = 1
        else:
            toffset = 0
        np_lines = []
        for j in range(n_j):
            aff_list = [int(i) - toffset for i in line.split()]  # Taillard spec has machines id start at 1
            np_lines.append(np.array(aff_list))
            line = next(f, "")
            if line == "":
                break
        affectations = np.stack(np_lines)

        return n_j, n_m, affectations, durations
