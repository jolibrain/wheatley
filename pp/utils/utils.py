import tqdm
from generic.graphgym.async_vector_env import AsyncGraphVectorEnv
from pp.graph.graph_factory import GraphFactory


def create_env(
    env_cls,
    problem_description,
    env_specification,
    i,
    lappe,
    rwpe,
    seed,
    create_new_pbs,
    env_kwargs=None,
):
    if env_kwargs is None:
        env_kwargs = {}

    def _init():
        env = env_cls(
            problem_description,
            env_specification,
            i,
            validate=False,
            lappe=lappe,
            rwpe=rwpe,
            seed=seed,
            create_new_pbs=create_new_pbs,
            **env_kwargs,
        )
        return env

    return _init


def pb_ids(problem_description, num_envs):
    if not hasattr(problem_description, "train_pbs"):
        return list(range(num_envs))  # simple env id
    # for psps, we should return a list per env of list of problems for this env
    if problem_description.unload:
        return [list(range(len(problem_description.train_pbs_ids)))] * num_envs
    else:
        return [list(range(len(problem_description.train_pbs)))] * num_envs


def create_train_envs(
    env_cls,
    problem_description,
    env_specification,
    num_envs,
    max_shared_mem_per_worker,
    lappe,
    rwpe,
    create_new_pbs,
    env_kwargs=None,
):
    pbs_per_env = pb_ids(problem_description, num_envs)

    envs = AsyncGraphVectorEnv(
        [
            create_env(
                env_cls,
                problem_description,
                env_specification,
                pbs_per_env[i],
                lappe,
                rwpe,
                n,
                create_new_pbs,
                env_kwargs=env_kwargs,
            )
            for n, i in enumerate(
                tqdm.tqdm(range(num_envs), desc="Creating learning envs")
            )
        ],
        # spwan helps when observation space is huge
        # and also with torch in subprocesses
        context="spawn",
        copy=False,
        shared_memory=True,
        disk=False,
        max_mem_size=max_shared_mem_per_worker,
        graph_factory=GraphFactory,
    )
    return envs
