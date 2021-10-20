import os

import numpy as np
import torch

from models.agent import Agent
from problem.problem_description import ProblemDescription
from utils.utils import generate_problem

from config import MAX_DURATION, MAX_N_JOBS, MAX_N_MACHINES
from args import args, exp_name


def main():

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.n_j > MAX_N_JOBS or args.n_m > MAX_N_MACHINES:
        raise Exception("MAX_N_JOBS or MAX_N_MACHINES are too low for this setup")

    print(
        f"Launching training\n"
        f"Problem size : {args.n_j} jobs, {args.n_m} machines\n"
        f"Transition model : {args.transition_model_config}\n"
        f"Reward model : {args.reward_model_config}\n"
        f"Seed : {args.seed}\n"
        f"Agent graph : {args.gconv_type}\n"
        f"Graph pooling : {args.graph_pooling}\n"
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

    path = "saved_networks/" + exp_name + ".zip" if args.path == "saved_networks/default_net" else args.path + ".zip"
    if args.retrain and os.path.exists(path):
        agent = Agent.load(
            path,
            args.input_list,
            args.add_force_insert_boolean,
            args.slot_locking,
            args.mlp_act,
            args.full_force_insert,
        )
    else:
        n_features = 2 + len(args.input_list)
        if "one_hot_job_id" in args.input_list:
            n_features += MAX_N_JOBS - 1
        if "one_hot_machine_id" in args.input_list:
            n_features += MAX_N_MACHINES - 1

        agent = Agent(
            n_epochs=args.n_epochs,
            n_steps_episode=args.n_steps_episode,
            batch_size=args.batch_size,
            gamma=args.gamma,
            clip_range=args.clip_range,
            target_kl=args.target_kl,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            lr=args.lr,
            gconv_type=args.gconv_type,
            graph_has_relu=args.graph_has_relu,
            graph_pooling=args.graph_pooling,
            optimizer=args.optimizer,
            freeze_graph=args.freeze_graph,
            input_dim_features_extractor=n_features,
            add_force_insert_boolean=args.add_force_insert_boolean,
            slot_locking=args.slot_locking,
            mlp_act=args.mlp_act,
            n_workers=args.n_workers,
            device=torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu"),
            input_list=args.input_list,
        )

    agent.train(
        problem_description,
        total_timesteps=args.total_timesteps,
        n_test_env=args.n_test_env,
        eval_freq=args.eval_freq,
        normalize_input=not args.dont_normalize_input,
        display_env=exp_name,
        path=path,
        fixed_benchmark=args.fixed_benchmark,
        full_force_insert=args.full_force_insert,
        custom_heuristic_name=args.custom_heuristic_name,
    )


if __name__ == "__main__":
    main()
