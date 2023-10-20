import sys

from psp.env.genv import GEnv
from generic.utils import decode_mask
from psp.env.graphgym.async_vector_env import AsyncGraphVectorEnv
import torch
from collections import deque


def test_genv(problem_description_small, env_specification_small):
    env = GEnv(problem_description_small, env_specification_small, [0])
    obs, reward, done, _, info = env.step(0)
    assert done == False
    assert reward == 0
    print("info[mask]", info["mask"])
    assert torch.equal(
        info["mask"],
        torch.tensor([False, True, True, False, False, False, False, False]),
    )

    obs, reward, done, _, info = env.step(1)
    assert done == False
    assert reward == 0
    assert torch.equal(
        info["mask"],
        torch.tensor([False, False, True, False, False, False, False, False]),
    )

    obs, reward, done, _, info = env.step(2)
    assert done == False
    assert reward == 0
    assert torch.equal(
        info["mask"],
        torch.tensor([False, False, False, True, True, False, False, False]),
    )

    obs, reward, done, _, info = env.step(3)
    assert done == False
    assert reward == 0
    assert torch.equal(
        info["mask"],
        torch.tensor([False, False, False, False, True, False, False, False]),
    )

    obs, reward, done, _, info = env.step(4)
    assert done == False
    assert reward == 0
    assert torch.equal(
        info["mask"],
        torch.tensor([False, False, False, False, False, True, True, False]),
    )

    obs, reward, done, _, info = env.step(5)
    assert done == False
    assert reward == 0
    assert torch.equal(
        info["mask"],
        torch.tensor([False, False, False, False, False, False, True, False]),
    )

    obs, reward, done, _, info = env.step(6)
    assert done == False
    assert reward == 0
    assert torch.equal(
        info["mask"],
        torch.tensor([False, False, False, False, False, False, False, True]),
    )

    obs, reward, done, _, info = env.step(7)
    assert done
    # assert reward == -15 unnormalized
    # assert reward == -1.25 normalized final value
    assert reward == -15 / len(env.state.job_modes)
    assert torch.equal(
        info["mask"],
        torch.tensor([False, False, False, False, False, False, False, False]),
    )


def create_env(problem_description, env_specification, i):
    def _init():
        env = GEnv(problem_description, env_specification, i, validate=False)
        return env

    return _init


def pb_ids(problem_description, num_envs):
    if not hasattr(problem_description, "train_psps"):
        return list(range(num_envs))  # simple env id
    # for psps, we should return a list per env of list of problems for this env
    return [list(range(len(problem_description.train_psps)))] * num_envs


def test_graphenv(problem_description_small, env_specification_small):
    num_envs = 2
    pbs_per_env = pb_ids(problem_description_small, num_envs)
    envs = AsyncGraphVectorEnv(
        [
            create_env(
                problem_description_small,
                env_specification_small,
                pbs_per_env[i],
            )
            for i in range(num_envs)
        ],
        # spwan helps when observation space is huge
        # context="spawn",
        copy=False,
    )

    o, info = envs.reset()

    obs = []
    action_masks = torch.empty((10, 2, env_specification_small.max_n_nodes))
    dones = torch.empty((10, 2))
    action_masks = torch.empty((10, 2, env_specification_small.max_n_nodes))
    rewards = torch.empty((10, 2))
    ep_info_buffer = deque(maxlen=100)

    next_obs = o
    action_mask = decode_mask(info["mask"])
    next_done = torch.zeros(2)

    actions = []
    actions.append([0, 0])
    actions.append([1, 2])
    actions.append([2, 4])
    actions.append([3, 6])
    actions.append([4, 1])
    actions.append([5, 3])
    actions.append([6, 5])
    actions.append([7, 7])

    for step in range(8):
        print("STEP ", step)
        obs.append(next_obs)
        action_masks[step] = torch.tensor(action_mask)
        dones[step] = next_done

        print("actions", actions[step])
        next_obs, reward, done, _, info = envs.step(actions[step])
        print("next_obs", next_obs)
        print("reward", reward)
        print("done", done)
        print("info", info)
        action_mask = decode_mask(info["mask"])
        if "final_info" in info:
            for ep_info in info["final_info"]:
                if ep_info is not None:  # some episode may be finished and other not
                    ep_info_buffer.append(ep_info["episode"])
                    # self.ep_info_buffer.extend(
                    #     [ep_info["episode"] for ep_info in info["final_info"]]
                    # )

        rewards[step] = torch.tensor(reward).view(-1)
        next_done = torch.Tensor(done)
