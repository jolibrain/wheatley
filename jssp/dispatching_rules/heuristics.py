"""All those heuristics are implemented following their description
in this paper: https://arxiv.org/pdf/2010.12367.pdf.
"""
import numpy as np


def shortest_processing_time(
    durations: np.ndarray,
    affectations: np.ndarray,
    schedule: np.ndarray,
    candidate_ids: np.ndarray,
) -> int:
    """Candidates with lower processing time come first."""
    n_jobs = durations.shape[0]

    frontier_candidates = schedule[:, :-1].argmin(axis=1)
    processing_times = np.take_along_axis(
        durations, frontier_candidates.reshape(n_jobs, 1), axis=1
    )
    processing_times = processing_times.reshape(n_jobs)[candidate_ids]
    best_candidate_index = np.argmin(processing_times)
    return candidate_ids[best_candidate_index]


def most_work_remaining(
    durations: np.ndarray,
    affectations: np.ndarray,
    schedule: np.ndarray,
    candidate_ids: np.ndarray,
) -> int:
    remaining_work = durations.copy()
    remaining_work[schedule[:, :-1] != -1] = 0
    remaining_work = remaining_work.sum(axis=1)
    remaining_work = remaining_work[candidate_ids]

    best_candidate_index = np.argmax(remaining_work)
    return candidate_ids[best_candidate_index]


def minimum_ratio_to_work_remaining(
    durations: np.ndarray,
    affectations: np.ndarray,
    schedule: np.ndarray,
    candidate_ids: np.ndarray,
) -> int:
    remaining_work = durations.copy()
    remaining_work[schedule[:, :-1] != -1] = 0
    remaining_work = remaining_work.sum(axis=1)
    remaining_work = remaining_work[candidate_ids]

    work_done = durations.copy()
    work_done[schedule[:, :-1] == -1] = 0
    work_done = work_done.sum(axis=1)
    work_done = work_done[candidate_ids]

    ratio = work_done / remaining_work

    best_candidate_index = np.argmin(ratio)
    return candidate_ids[best_candidate_index]


def most_operations_remaining(
    durations: np.ndarray,
    affectations: np.ndarray,
    schedule: np.ndarray,
    candidate_ids: np.ndarray,
) -> int:
    operations_remaining = schedule[:, :-1] == -1
    operations_remaining = operations_remaining.sum(axis=1)
    operations_remaining = operations_remaining[candidate_ids]
    best_candidate_index = np.argmax(operations_remaining)
    return candidate_ids[best_candidate_index]


def random_rule(
    durations: np.ndarray,
    affectations: np.ndarray,
    schedule: np.ndarray,
    candidate_ids: np.ndarray,
) -> int:
    rng = np.random.default_rng(0)
    return rng.permutation(candidate_ids)[0]


HEURISTICS = {
    "SPT": shortest_processing_time,
    "MWKR": most_work_remaining,
    "MOPNR": most_operations_remaining,
    "FDD/MWKR": minimum_ratio_to_work_remaining,
    "random": random_rule,
}
