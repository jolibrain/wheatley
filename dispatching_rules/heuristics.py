"""All those heuristics are implemented following their description
in this paper: https://arxiv.org/pdf/2010.12367.pdf.
"""
import numpy as np


def shortest_processing_time(
    processing_times: np.ndarray,
    machines: np.ndarray,
    starting_times: np.ndarray,
    candidate_ids: np.ndarray,
) -> int:
    """Candidates with lower processing time come first."""
    n_candidates = candidate_ids.shape[0]
    n_jobs = processing_times.shape[0]

    frontier_candidates = starting_times[:, :-1].argmin(axis=1)
    processing_times = np.take_along_axis(
        processing_times, frontier_candidates.reshape(n_jobs, 1), axis=1
    )
    processing_times = processing_times.reshape(n_jobs)[candidate_ids]
    best_candidate_index = np.argmin(processing_times)
    return candidate_ids[best_candidate_index]


def most_work_remaining(
    processing_times: np.ndarray,
    machines: np.ndarray,
    starting_times: np.ndarray,
    candidate_ids: np.ndarray,
) -> int:
    remaining_work = processing_times[:, :-1].copy()
    remaining_work[remaining_work != -1] = 0
    remaining_work = remaining_work.sum(axis=1)
    remaining_work = remaining_work[candidate_ids]

    best_candidate_index = np.argmax(remaining_work)
    return candidate_ids[best_candidate_index]


def minimum_ratio_to_work_remaining(
    processing_times: np.ndarray,
    machines: np.ndarray,
    starting_times: np.ndarray,
    candidate_ids: np.ndarray,
) -> int:
    remaining_work = processing_times[:, :-1].copy()
    remaining_work[remaining_work != -1] = 0
    remaining_work = remaining_work.sum(axis=1)
    remaining_work = remaining_work[candidate_ids]

    work_done = processing_times[:, :-1].copy()
    work_done[work_done == -1] = 0
    work_done = work_done.sum(axis=1)
    work_done = work_done[candidate_ids]

    # Avoid division by 0.
    remaining_work[remaining_work == 0] = max(remaining_work.max(), 1)
    ratio = work_done / remaining_work

    best_candidate_index = np.argmin(ratio)
    return candidate_ids[best_candidate_index]


def most_operations_remaining(
    processing_times: np.ndarray,
    machines: np.ndarray,
    starting_times: np.ndarray,
    candidate_ids: np.ndarray,
) -> int:
    operations_remaining = starting_times[:, :-1] == -1
    operations_remaining = operations_remaining.sum(axis=1)
    operations_remaining = operations_remaining[candidate_ids]
    best_candidate_index = np.argmax(operations_remaining)
    return candidate_ids[best_candidate_index]


def random_rule(
    processing_times: np.ndarray,
    machines: np.ndarray,
    starting_times: np.ndarray,
    candidate_ids: np.ndarray,
) -> int:
    rng = np.random.default_rng(0)
    return rng.permutation(candidate_ids)[0]


HEURISTICS = {
    "SFT": shortest_processing_time,
    "MWKR": most_work_remaining,
    "MOPNR": most_operations_remaining,
    "FDD/MWKR": minimum_ratio_to_work_remaining,
    "random": random_rule,
}
