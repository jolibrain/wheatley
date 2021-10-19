# running benchmark on Taillard's problems

import sys
import glob
import os

import torch

sys.path.append('..')
from utils.utils import load_taillard_problem
from problem.problem_description import ProblemDescription
from config import MAX_DURATION, MAX_N_JOBS, MAX_N_MACHINES
from args import args, exp_name, parser
from models.agent import Agent
from utils.utils_testing import test_agent

# additional options
parser.add_argument('--taillard_pbs', help='taillard problem name (e.g ta01), default is empty for benchmarking all problems', default='*')
args = parser.parse_args()
#print('done with parsing')

# get Taillard problem files
all_problem_files = glob.glob('../instances/taillard/' + args.taillard_pbs + '.txt')

args.fixed_problem = True

def main():

    # iterate problems
    taillards = {}
    for pr_file in all_problem_files:
        # load problem
        taillard_pb = os.path.basename(pr_file).replace('.txt','')
        n_j, n_m, affectations, durations = load_taillard_problem(pr_file)

        # update arguments
        args.n_j = n_j
        args.n_m = n_m
        args.n_steps_episode = n_j * n_m * 5
        args.batch_size = n_j * n_m
        args.n_test_env = 1
        exp_name = (
            f"{args.n_j}j{args.n_m}m_{args.transition_model_config}_{args.reward_model_config}_{args.gconv_type}_taillard_{taillard_pb}"
    )

        if args.remove_machine_id:
            exp_name += "_RMI"
        if args.fixed_benchmark:
            exp_name += "_FB"
        if args.dont_normalize_input:
            exp_name += "_DNI"
        if args.freeze_graph:
            exp_name += "_FG"
        if args.one_hot_machine_id:
            exp_name += "_OHMI"
        if args.add_force_insert_boolean:
            exp_name += "_FI"
        if args.slot_locking:
            exp_name += "_SL"
        if args.exp_name_appendix is not None:
            exp_name += "_" + args.exp_name_appendix
        if args.max_pool:
            exp_name += "_max"

        problem_description = ProblemDescription(n_j, n_m, MAX_DURATION,
                                                 args.transition_model_config,
                                                 args.reward_model_config,
                                                 affectations,
                                                 durations)

        # solve for problem
        if args.remove_machine_id:
                input_dim_features_extractor = 2
        else:
            if args.one_hot_machine_id:
                input_dim_features_extractor = 2 + MAX_N_MACHINES
            else:
                input_dim_features_extractor = 3
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
                max_pool=args.max_pool,
                optimizer=args.optimizer,
                add_machine_id=not args.remove_machine_id,
                freeze_graph=args.freeze_graph,
                input_dim_features_extractor=input_dim_features_extractor,
                one_hot_machine_id=args.one_hot_machine_id,
                add_force_insert_boolean=args.add_force_insert_boolean,
                slot_locking=args.slot_locking,
                mlp_act=args.mlp_act,
                n_workers=args.n_workers,
                device=torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu"),
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
        )

        # test agent
        print("testing agent")
        agent = Agent.load(path, not args.remove_machine_id, args.one_hot_machine_id, args.add_pdr_boolean)
        rl_makespan = test_agent(agent, problem_description)

        # store best metric
        print('Taillard ',taillard_pb, ' / best makespace=',rl_makespan)
        taillards[taillard_pb] = rl_makespan

    # print results
    print('Taillard benchmark results')
    print(taillards)

if __name__ == "__main__":
    main()
