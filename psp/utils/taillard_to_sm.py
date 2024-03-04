from itertools import product
from pathlib import Path

import numpy as np


def convert_to_sm(durations: np.ndarray, affectations: np.ndarray) -> str:
    """Convert a JSSP instance to a SM RCPSP file.

    The JSSP machines are represented as differents renewable resources with a capacity of 1.
    The JSSP tasks have only one successor (the successor in its corresponding JSSP job).
    The JSSP tasks consume 1 resource unit of the machine they are assigned to.
    A source and sink node is added. The source has as successors all the first tasks of each job.
    The sink is the successor of all the last tasks of each job.


    ---
    Args:
        durations: The duration of each task on each machine.
            Shape of [n_jobs, n_machines].
        affectations: The machine on which each task is executed.
            Shape of [n_jobs, n_machines].

    ---
    Returns:
        The content of the SM file as a string.
    """
    n_jobs, n_machines = durations.shape
    n_tasks = n_jobs * n_machines + 2  # Include source and sink.

    assert np.all(affectations != -1), "This script does not support fictive tasks."

    if affectations.max() == n_machines:
        print("Machines are detected to be in the range [1, n_machines].")
        affectations = affectations - 1

    stars = "*" * 72 + "\n"
    filecontent = ""

    filecontent += stars
    filecontent += "file with basedata            : ?\n"
    filecontent += "initial value random generator: ?\n"

    filecontent += stars
    filecontent += "projects                      :  1\n"
    filecontent += f"jobs (incl. supersource/sink ):  {n_tasks}\n"
    filecontent += "horizon                       :  ?\n"
    filecontent += "RESOURCES\n"
    filecontent += f"  - renewable                 :  {n_machines}   R\n"
    filecontent += "  - nonrenewable              :  0   N\n"
    filecontent += "  - doubly constrained        :  0   D\n"

    filecontent += stars
    filecontent += "PROJECT INFORMATION:\n"
    filecontent += "pronr.  #jobs rel.date duedate tardcost  MPM-Time\n"
    filecontent += f"    1     {n_tasks - 2}      0       ?       ?       ?\n"

    ### PRECEDENCES ###
    filecontent += stars
    filecontent += "PRECEDENCE RELATIONS:\n"
    filecontent += "jobnr.    #modes  #successors   successors\n"
    mode = 1

    # Source.
    source_id, sink_id = 1, n_tasks
    starting_tasks = "  ".join(f"{i * n_machines + 2}" for i in range(n_jobs))
    filecontent += f"{source_id}      {mode}      {n_jobs}      {starting_tasks}\n"

    # Tasks.
    for job_id, machine_id in product(range(n_jobs), range(n_machines)):
        # Tasks id starts at 1, and the first task is taken by the source node.
        task_id = job_id * n_machines + machine_id + 2
        successor = sink_id if machine_id == (n_machines - 1) else task_id + 1
        filecontent += f"{task_id}      {mode}      {1}      {successor}\n"

    # Sink.
    filecontent += f"{sink_id}      {mode}      {0}      \n"

    ### DURATIONS ###
    filecontent += stars
    filecontent += "REQUESTS/DURATIONS:\n"
    filecontent += (
        "jobnr. mode duration  "
        + "  ".join(f"R {resource_id}" for resource_id in range(1, n_machines + 1))
        + "\n"
    )
    filecontent += stars.replace("*", "-")

    # Source.
    filecontent += (
        f"{source_id}      {mode}      {0}      "
        + "   ".join("0" for _ in range(n_machines))
        + "\n"
    )

    # Tasks.
    for job_id, machine_id in product(range(n_jobs), range(n_machines)):
        # Tasks id starts at 1, and the first task is taken by the source node.
        task_id = job_id * n_machines + machine_id + 2
        duration = durations[job_id, machine_id]
        resource_id = affectations[job_id, machine_id]
        filecontent += (
            f"{task_id}      {mode}      {duration}      "
            + "   ".join("1" if resource_id == m else "0" for m in range(n_machines))
            + "\n"
        )

    # Sink.
    filecontent += (
        f"{sink_id}      {mode}      {0}      "
        + "   ".join("0" for _ in range(n_machines))
        + "\n"
    )

    ### RESOURCE AVAILABILITIES ###
    filecontent += stars
    filecontent += "RESOURCEAVAILABILITIES:\n"
    filecontent += (
        "   "
        + "   ".join(f"R {resource_id}" for resource_id in range(1, n_machines + 1))
        + "\n"
    )
    filecontent += "   " + "   ".join("1" for _ in range(1, n_machines + 1)) + "\n"
    filecontent += stars.replace("\n", "")

    return filecontent


def convert_all_taillards(taillard_dir: Path, psp_dir: Path):
    from jssp.env.state import State

    psp_dir.mkdir(parents=True, exist_ok=True)

    for instances_dir in taillard_dir.glob("*x*"):
        n_jobs, n_machines = str(instances_dir.name).split("x")
        n_jobs, n_machines = int(n_jobs), int(n_machines)
        (psp_dir / instances_dir.name).mkdir(exist_ok=True)
        for instance_path in instances_dir.glob("*.npz"):
            state = State.from_instance_file(
                instance_path,
                n_jobs,
                n_machines,
                6 + n_machines,
                deterministic=True,
                feature_list=[],
                observe_conflicts_as_cliques=False,
            )
            durations = state.original_durations[:, :, 0]
            affectations = state.affectations

            filecontent = convert_to_sm(durations, affectations)

            filepath = psp_dir / instances_dir.name / instance_path.name
            filepath = filepath.with_suffix(".sm")
            with open(filepath, "w") as sm_file:
                sm_file.write(filecontent)


if __name__ == "__main__":
    # Launch with `python3 -m psp.utils.taillard_to_sm`.

    convert_all_taillards(
        Path("./instances/jssp/deterministic/"),
        Path("./instances/psp/taillards/"),
    )
