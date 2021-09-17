import numpy as np
import torch

from models.agent import Agent
from problem.problem_description import ProblemDescription
from utils.utils import generate_problem

from config import MAX_DURATION, MAX_N_JOBS, MAX_N_MACHINES
from args import args


def main():

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.n_j > MAX_N_JOBS or args.n_m > MAX_N_MACHINES:
        raise Exception("MAX_N_JOBS or MAX_N_MACHINES are too low for this setup")

    print(
        f"Launching training\n"
        f"Problem size : {args.n_j} jobs, {args.n_m} machines\n"
        f"Training time : {args.total_timesteps} timesteps"
    )

    if args.fixed_problem:
        affectations, durations = generate_problem(args.n_j, args.n_m, MAX_DURATION)
        print(affectations)
        print(durations)
    else:
        affectations, durations = None, None
    problem_description = ProblemDescription(
        args.n_j,
        args.n_m,
        MAX_DURATION,
        args.transition_model_config,
        args.reward_model_config,
        affectations,
        durations,
    )

    agent = Agent(
        n_epochs=args.n_epochs,
        n_steps_episode=args.n_steps_episode,
        batch_size=args.batch_size,
        gamma=args.gamma,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        lr=args.lr,
        add_machine_id=args.add_machine_id,
    )
    agent.train(
        problem_description,
        total_timesteps=args.total_timesteps,
        n_test_env=args.n_test_env,
        eval_freq=args.eval_freq,
        divide_loss=not args.dont_divide_loss,
        display_env=f"{args.n_j}j{args.n_m}m{args.total_timesteps}steps{args.seed}seed",
        n_workers=args.n_workers,
        multiprocessing=args.multiprocessing,
        path=args.path,
        fixed_benchmark=args.fixed_benchmark,
    )


if __name__ == "__main__":
    main()
