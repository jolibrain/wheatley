from psp.utils.loaders import PSPLoader
from psp.models.agent import Agent
from psp.env.genv import GEnv
from generic.utils import decode_mask
from psp.description import Description
import os


def solve_instance(problem_description, agent):
    venv = GEnv(
        problem_description, agent.env_specification, [0], validate=True, pyg=True
    )
    obs, info = venv.reset(soft=True)
    done = False
    while not done:
        action_masks = info["mask"].reshape(1, -1)
        action_masks = decode_mask(action_masks)
        obs = agent.obs_as_tensor_add_batch_dim(obs)
        action = agent.predict(obs, deterministic=True, action_masks=action_masks)
        obs, reward, done, _, info = venv.step(action.long().item())
    sol = venv.get_solution()

    return sol


if __name__ == "__main__":
    from args import argument_parser, parse_args

    parser = argument_parser()
    args, _ = parse_args(parser)

    assert (
        args.load_problem is not None
    ), "You should provide a problem to solve (use --load_problem)."

    agent = Agent.load(args.path, max_n_modes=args.max_n_modes)
    loader = PSPLoader()

    psp = loader.load_single(args.load_problem)
    train_psps = [psp]
    test_psps = [psp]
    problem_description = Description(
        transition_model_config=args.transition_model_config,
        reward_model_config=args.criterion,
        deterministic=(args.duration_type == "deterministic"),
        train_psps=train_psps,
        test_psps=test_psps,
        seed=args.seed,
    )
    agent.to(args.device)
    sol = solve_instance(problem_description, agent)
    sol.save(os.path.basename(psp.pb_id) + ".sol", psp.pb_id)
