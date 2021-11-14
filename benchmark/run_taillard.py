# running benchmark on Taillard's problems

import sys
import glob
import os

import torch

sys.path.append("..")

from utils.utils import load_taillard_problem  # noqa E402
from problem.problem_description import ProblemDescription  # noqa E402
from config import MAX_DURATION, MAX_N_JOBS, MAX_N_MACHINES  # noqa E402
from args import parser  # noqa E402
from models.agent import Agent  # noqa E402
from utils.utils_testing import test_agent  # noqa E402


# additional options
parser.add_argument(
    "--taillard_pbs", help="taillard problem name (e.g ta01), default is empty for benchmarking all problems", default="*"
)
args = parser.parse_args()
# print('done with parsing')

# get Taillard problem files
all_problem_files = glob.glob("../instances/taillard/" + args.taillard_pbs + ".txt")

args.fixed_problem = True
args.input_list = args.features


def main():

    # iterate problems
    taillards = {}
    for pr_file in all_problem_files:
        # load problem
        taillard_pb = os.path.basename(pr_file).replace(".txt", "")
        n_j, n_m, affectations, durations = load_taillard_problem(pr_file)

        if n_j > MAX_N_JOBS or n_m > MAX_N_MACHINES:
            raise Exception("MAX_N_JOBS or MAX_N_MACHINE is too low for this setup")

        # update arguments
        args.n_j = n_j
        args.n_m = n_m
        args.n_steps_episode = n_j * n_m * 5
        args.batch_size = n_j * n_m
        args.n_test_env = 1
        exp_name = f"{args.n_j}j{args.n_m}m_{args.transition_model_config}_{args.reward_model_config}_{args.gconv_type}_{args.graph_pooling}_taillard_{taillard_pb}"

        if args.fixed_benchmark:
            exp_name += "_FB"
        if args.dont_normalize_input:
            exp_name += "_DNI"
        if args.freeze_graph:
            exp_name += "_FG"
        if args.add_force_insert_boolean:
            exp_name += "_FI"
        if args.slot_locking:
            exp_name += "_SL"
        if args.exp_name_appendix is not None:
            exp_name += "_" + args.exp_name_appendix

        problem_description = ProblemDescription(
            n_j, n_m, MAX_DURATION, args.transition_model_config, args.reward_model_config, affectations, durations
        )

        # solve for problem
        n_features = 2 + len(args.input_list)
        if "one_hot_job_id" in args.input_list:
            n_features += MAX_N_JOBS - 1
        if "one_hot_machine_id" in args.input_list:
            n_features += MAX_N_MACHINES - 1

        path = "saved_networks/" + exp_name + ".zip" if args.path == "saved_networks/default_net" else args.path + ".zip"

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
            ortools_strategy = args.ortools_strategy,
            keep_same_testing_envs = not args.change_testing_envs
        )

        # test agent
        print("testing agent")
        agent = Agent.load(
            path,
            args.input_set,
            args.add_force_insert_boolean,
            args.slot_locking,
            args.mlp_act,
            args.n_workers,
            torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu"),
        )
        rl_makespan = test_agent(agent, problem_description)

        # store best metric
        print("Taillard ", taillard_pb, " / best makespace=", rl_makespan)
        taillards[taillard_pb] = rl_makespan

    # print results
    print("Taillard benchmark results")
    print(taillards)


if __name__ == "__main__":
    main()
