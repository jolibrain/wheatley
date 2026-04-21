"""An async vector environment."""

import os

import sys
from copy import deepcopy
from enum import Enum
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
import pickle
import io
import contextlib
import torch.multiprocessing as mp
from psp.graph.graph_factory import GraphFactory
import numpy as np
import time
import torch
import pickle

# import tracemalloc

from psp.env.genv import GEnv as Env

from .vector_env import GraphVectorEnv

__all__ = ["AsyncVectorEnv"]


class AsyncState(Enum):
    DEFAULT = "default"
    WAITING_RESET = "reset"
    WAITING_STEP = "step"
    WAITING_CALL = "call"


class CloudpickleWrapper:
    """Wrapper that uses cloudpickle to pickle and unpickle the result."""

    def __init__(self, fn: Callable[[], Env]):
        """Cloudpickle wrapper for a function."""
        self.fn = fn

    def __getstate__(self):
        """Get the state using `cloudpickle.dumps(self.fn)`."""
        import cloudpickle

        return cloudpickle.dumps(self.fn)

    def __setstate__(self, ob):
        """Sets the state with obs."""
        import pickle

        self.fn = pickle.loads(ob)

    def __call__(self):
        """Calls the function `self.fn` with no arguments."""
        return self.fn()


def create_shared_memory(maxsize, n, ctx, disk):
    if disk:
        fnames = []
        for i in range(n):
            fname = "/tmp/wheatley_" + str(os.getpid()) + "_worker_" + str(i) + ".obs"
            fnames.append(fname)

        return fnames
    else:
        return ([ctx.Array("B", maxsize) for i in range(n)], ctx.Array("Q", n))


def read_from_shared_memory(shared_memory, n, disk, pyg):
    if disk:
        # return [dgl.load_graphs(shared_memory[i])[0][0] for i in range(n)]
        return [GraphFactory.load(shared_memory[i], pyg) for i in range(n)]
    else:
        # return [
        #     torch.load(
        #         io.BytesIO(bytearray(shared_memory[0][i][: shared_memory[1][i]]))
        #     )
        #     for i in range(n)
        # ]
        return [
            GraphFactory.deserialize(
                bytearray(shared_memory[0][i][: shared_memory[1][i]]), pyg
            )
            for i in range(n)
        ]


def write_to_shared_memory(index, obs, shared_memory, disk, pyg, max_mem_size):
    if disk:
        obs.save(shared_memory[index])
    else:
        # buf = io.BytesIO()
        # torch.save(obs, buf, pickle_protocol=pickle.HIGHEST_PROTOCOL)
        # out = buf.getvalue()
        # l = len(out)
        out, l = obs.serialize()
        if l > max_mem_size:
            print(f"insufficient shared memory size : {max_mem_size}   needed: {l}")
            exit(1)
        shared_memory[0][index][:l] = out
        shared_memory[1][index] = l


class AsyncGraphVectorEnv(GraphVectorEnv):
    """Vectorized environment that runs multiple environments in parallel.

    It uses ``multiprocessing`` processes, and pipes for communication.

    Example:
        >>> import gymnasium as gym
        >>> env = gym.vector.AsyncVectorEnv([
        ...     lambda: gym.make("Pendulum-v1", g=9.81),
        ...     lambda: gym.make("Pendulum-v1", g=1.62)
        ... ])
        >>> env.reset(seed=42)
        (array([[-0.14995256,  0.9886932 , -0.12224312],
               [ 0.5760367 ,  0.8174238 , -0.91244936]], dtype=float32), {})
    """

    def __init__(
        self,
        env_fns: Sequence[Callable[[], Env]],
        shared_memory: bool = False,
        copy: bool = True,
        context: Optional[str] = None,
        daemon: bool = True,
        worker: Optional[Callable] = None,
        disk=True,
        pyg=True,
        max_mem_size=2000000,
    ):
        super().__init__(
            num_envs=len(env_fns),
        )
        ctx = mp.get_context(context)
        mp.set_sharing_strategy("file_system")
        self.env_fns = env_fns
        self.shared_memory = shared_memory
        self.disk = disk
        self.copy = copy
        dummy_env = env_fns[0]()
        dummy_env.close()
        del dummy_env
        self.pyg = pyg

        if self.shared_memory:
            self._obs_buffer = create_shared_memory(
                max_mem_size, n=self.num_envs, ctx=ctx, disk=self.disk
            )
        else:
            _obs_buffer = None
        self.observations = []

        self.parent_pipes, self.processes = [], []
        self.error_queue = ctx.Queue()
        target = _worker_shared_memory if self.shared_memory else _worker
        target = worker or target
        for idx, env_fn in enumerate(self.env_fns):
            parent_pipe, child_pipe = ctx.Pipe()
            process = ctx.Process(
                target=target,
                name=f"Worker<{type(self).__name__}>-{idx}",
                daemon=True,
                args=(
                    idx,
                    CloudpickleWrapper(env_fn),
                    child_pipe,
                    parent_pipe,
                    self._obs_buffer,
                    self.disk,
                    self.error_queue,
                    self.pyg,
                    max_mem_size,
                ),
            )

            self.parent_pipes.append(parent_pipe)
            self.processes.append(process)

            process.daemon = daemon
            process.start()
            child_pipe.close()

        self._state = AsyncState.DEFAULT

    def reset_async(
        self,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[dict] = None,
    ):
        self._assert_is_running()

        if seed is None:
            seed = [None for _ in range(self.num_envs)]
        if isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        assert len(seed) == self.num_envs

        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                f"Calling `reset_async` while waiting for a pending call to `{self._state.value}` to complete",
                self._state.value,
            )

        for pipe, single_seed in zip(self.parent_pipes, seed):
            single_kwargs = {}
            if single_seed is not None:
                single_kwargs["seed"] = single_seed
            if options is not None:
                single_kwargs["options"] = options

            pipe.send(("reset", single_kwargs))
        self._state = AsyncState.WAITING_RESET

    def reset_wait(
        self,
        timeout: Optional[Union[int, float]] = None,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        self._assert_is_running()
        if self._state != AsyncState.WAITING_RESET:
            raise NoAsyncCallError(
                "Calling `reset_wait` without any prior " "call to `reset_async`.",
                AsyncState.WAITING_RESET.value,
            )

        if not self._poll(timeout):
            self._state = AsyncState.DEFAULT
            raise mp.TimeoutError(
                f"The call to `reset_wait` has timed out after {timeout} second(s)."
            )

        observations_list, infos = [], {}
        successes = []

        for i, pipe in enumerate(self.parent_pipes):
            result, success = pipe.recv()
            successes.append(success)
            if success:
                obs, info = result
                if not self.shared_memory:
                    observations_list.append(obs)
                infos = self._add_info(infos, info, i)

        self._raise_if_errors(successes)
        self._state = AsyncState.DEFAULT

        if not self.shared_memory:
            self.observations = observations_list
        else:
            self.observations = read_from_shared_memory(
                self._obs_buffer, n=self.num_envs, disk=self.disk, pyg=self.pyg
            )

        return (deepcopy(self.observations) if self.copy else self.observations), infos

    def step_async(self, actions):
        self._assert_is_running()
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                f"Calling `step_async` while waiting for a pending call to `{self._state.value}` to complete.",
                self._state.value,
            )

        for n, pipe in enumerate(self.parent_pipes):
            pipe.send(("step", actions[n]))
        self._state = AsyncState.WAITING_STEP

    def step_wait(self, timeout: Optional[Union[int, float]] = None):
        self._assert_is_running()
        if self._state != AsyncState.WAITING_STEP:
            raise NoAsyncCallError(
                "Calling `step_wait` without any prior call " "to `step_async`.",
                AsyncState.WAITING_STEP.value,
            )

        if not self._poll(timeout):
            self._state = AsyncState.DEFAULT
            raise mp.TimeoutError(
                f"The call to `step_wait` has timed out after {timeout} second(s)."
            )

        observations_list, rewards, terminateds, truncateds, infos = [], [], [], [], {}
        successes = []
        for i, pipe in enumerate(self.parent_pipes):
            result, success = pipe.recv()
            successes.append(success)
            if success:
                obs, rew, terminated, truncated, info = result
                if not self.shared_memory:
                    observations_list.append(obs)
                rewards.append(rew)
                terminateds.append(terminated.item())
                truncateds.append(truncated)
                infos = self._add_info(infos, info, i)

        self._raise_if_errors(successes)
        self._state = AsyncState.DEFAULT

        if not self.shared_memory:
            self.observations = observations_list
        else:
            self.observations = read_from_shared_memory(
                self._obs_buffer, n=self.num_envs, disk=self.disk, pyg=self.pyg
            )

        return (
            deepcopy(self.observations) if self.copy else self.observations,
            np.array(rewards),
            np.array(terminateds, dtype=np.bool_),
            np.array(truncateds, dtype=np.bool_),
            infos,
        )

    def call_async(self, name: str, *args, **kwargs):
        self._assert_is_running()
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                "Calling `call_async` while waiting "
                f"for a pending call to `{self._state.value}` to complete.",
                self._state.value,
            )

        for pipe in self.parent_pipes:
            pipe.send(("_call", (name, args, kwargs)))
        self._state = AsyncState.WAITING_CALL

    def call_wait(self, timeout: Optional[Union[int, float]] = None) -> list:
        self._assert_is_running()
        if self._state != AsyncState.WAITING_CALL:
            raise NoAsyncCallError(
                "Calling `call_wait` without any prior call to `call_async`.",
                AsyncState.WAITING_CALL.value,
            )

        if not self._poll(timeout):
            self._state = AsyncState.DEFAULT
            raise mp.TimeoutError(
                f"The call to `call_wait` has timed out after {timeout} second(s)."
            )

        results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)
        self._state = AsyncState.DEFAULT

        return results

    def set_attr(self, name: str, values: Union[list, tuple, object]):
        self._assert_is_running()
        if not isinstance(values, (list, tuple)):
            values = [values for _ in range(self.num_envs)]
        if len(values) != self.num_envs:
            raise ValueError(
                "Values must be a list or tuple with length equal to the "
                f"number of environments. Got `{len(values)}` values for "
                f"{self.num_envs} environments."
            )

        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                "Calling `set_attr` while waiting "
                f"for a pending call to `{self._state.value}` to complete.",
                self._state.value,
            )

        for pipe, value in zip(self.parent_pipes, values):
            pipe.send(("_setattr", (name, value)))
        _, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)

    def close_extras(
        self, timeout: Optional[Union[int, float]] = None, terminate: bool = False
    ):
        timeout = 0 if terminate else timeout
        try:
            if self._state != AsyncState.DEFAULT:
                print(
                    f"Calling `close` while waiting for a pending call to `{self._state.value}` to complete."
                )
                function = getattr(self, f"{self._state.value}_wait")
                function(timeout)
        except mp.TimeoutError:
            terminate = True

        if terminate:
            for process in self.processes:
                if process.is_alive():
                    process.terminate()
        else:
            for pipe in self.parent_pipes:
                if (pipe is not None) and (not pipe.closed):
                    pipe.send(("close", None))
            for pipe in self.parent_pipes:
                if (pipe is not None) and (not pipe.closed):
                    pipe.recv()

        for pipe in self.parent_pipes:
            if pipe is not None:
                pipe.close()
        for process in self.processes:
            process.join()

    def _poll(self, timeout=None):
        self._assert_is_running()
        if timeout is None:
            return True
        end_time = time.perf_counter() + timeout
        delta = None
        for pipe in self.parent_pipes:
            delta = max(end_time - time.perf_counter(), 0)
            if pipe is None:
                return False
            if pipe.closed or (not pipe.poll(delta)):
                return False
        return True

    def _assert_is_running(self):
        if self.closed:
            raise ClosedEnvironmentError(
                f"Trying to operate on `{type(self).__name__}`, after a call to `close()`."
            )

    def _raise_if_errors(self, successes):
        if all(successes):
            return

        num_errors = self.num_envs - sum(successes)
        assert num_errors > 0
        for i in range(num_errors):
            index, exctype, value = self.error_queue.get()
            print(
                f"Received the following error from Worker-{index}: {exctype.__name__}: {value}"
            )
            print(f"Shutting down Worker-{index}.")
            self.parent_pipes[index].close()
            self.parent_pipes[index] = None

            if i == num_errors - 1:
                print("Raising the last exception back to the main process.")
                raise exctype(value)

    def __del__(self):
        """On deleting the object, checks that the vector environment is closed."""
        if self.shared_memory and self.disk:
            for b in self._obs_buffer:
                try:
                    os.remove(b)
                except:
                    pass

        if not getattr(self, "closed", True) and hasattr(self, "_state"):
            self.close(terminate=True)


def _worker(index, env_fn, pipe, parent_pipe, shared_memory, disk, error_queue):
    assert shared_memory is None
    env = env_fn()
    parent_pipe.close()
    try:
        while True:
            command, data = pipe.recv()
            if command == "reset":
                observation, info = env.reset(**data)
                pipe.send(((observation, info), True))

            elif command == "step":
                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(data)
                if terminated or truncated:
                    old_observation, old_info = observation, info
                    observation, info = env.reset()
                    info["final_observation"] = old_observation
                    info["final_info"] = old_info
                pipe.send(((observation, reward, terminated, truncated, info), True))
            elif command == "seed":
                env.seed(data)
                pipe.send((None, True))
            elif command == "close":
                pipe.send((None, True))
                break
            elif command == "_call":
                name, args, kwargs = data
                if name in ["reset", "step", "seed", "close"]:
                    raise ValueError(
                        f"Trying to call function `{name}` with "
                        f"`_call`. Use `{name}` directly instead."
                    )
                function = getattr(env, name)
                if callable(function):
                    pipe.send((function(*args, **kwargs), True))
                else:
                    pipe.send((function, True))
            elif command == "_setattr":
                name, value = data
                setattr(env, name, value)
                pipe.send((None, True))
            else:
                raise RuntimeError(
                    f"Received unknown command `{command}`. Must "
                    "be one of {`reset`, `step`, `seed`, `close`, `_call`, "
                    "`_setattr`}."
                )
    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()


def _worker_shared_memory(
    index,
    env_fn,
    pipe,
    parent_pipe,
    shared_memory,
    disk,
    error_queue,
    pyg,
    max_mem_size,
):
    assert shared_memory is not None
    env = env_fn()
    parent_pipe.close()
    mp.set_sharing_strategy("file_system")

    # tracemalloc.start()
    try:
        while True:
            command, data = pipe.recv()
            if command == "reset":
                # snap1 = tracemalloc.take_snapshot()
                observation, info = env.reset(**data)
                write_to_shared_memory(
                    index, observation, shared_memory, disk, pyg, max_mem_size
                )
                pipe.send(((None, info), True))
                # snap2 = tracemalloc.take_snapshot()
                # top_stats = snap2.compare_to(snap1, "lineno")
                # for stat in top_stats:
                #     print("reset stat ", stat)

            elif command == "step":
                # print("getting mem snapshot"n)
                # snap1 = tracemalloc.take_snapshot()
                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(data)
                if terminated or truncated:
                    old_observation, old_info = observation, info
                    observation, info = env.reset()
                    info["final_observation"] = old_observation
                    info["final_info"] = old_info

                write_to_shared_memory(
                    index, observation, shared_memory, disk, pyg, max_mem_size
                )
                pipe.send(((None, reward, terminated, truncated, info), True))
                # snap2 = tracemalloc.take_snapshot()
                # top_stats = snap2.compare_to(snap1, "lineno")
                # for stat in top_stats:
                #     print("step stat ", stat)
            elif command == "seed":
                env.seed(data)
                pipe.send((None, True))
            elif command == "close":
                pipe.send((None, True))
                break
            elif command == "_call":
                name, args, kwargs = data
                if name in ["reset", "step", "seed", "close"]:
                    raise ValueError(
                        f"Trying to call function `{name}` with "
                        f"`_call`. Use `{name}` directly instead."
                    )
                function = getattr(env, name)
                if callable(function):
                    pipe.send((function(*args, **kwargs), True))
                else:
                    pipe.send((function, True))
            elif command == "_setattr":
                name, value = data
                setattr(env, name, value)
                pipe.send((None, True))
            else:
                raise RuntimeError(
                    f"Received unknown command `{command}`. Must "
                    "be one of {`reset`, `step`, `seed`, `close`, `_call`, "
                    "`_setattr`}."
                )
    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        # tracemalloc.stop()
        env.close()
