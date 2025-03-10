from psp.utils.loaders import PSPLoader
from psp.models.agent import Agent
from psp.env.genv import GEnv
from generic.utils import decode_mask
from psp.description import Description
import os
import tqdm
import torch


def solve_instance(problem_description, agent):
    print("creating inference env")
    venv = GEnv(
        problem_description,
        agent.env_specification,
        [0],
        validate=True,
        pyg=True,
        reset=False,
    )
    print("reseting inference env")
    obs, info = venv.reset(soft=False)
    done = False
    for _ in tqdm.tqdm(range(problem_description.test_psps[0].n_jobs)):
        action_masks = info["mask"].reshape(1, -1)
        action_masks = decode_mask(action_masks)
        obs = agent.obs_as_tensor_add_batch_dim(obs)
        action = agent.predict(
            agent.preprocess(obs), deterministic=True, action_masks=action_masks
        )
        obs, reward, done, _, info = venv.step(action.long().item())
        if done:
            break
    sol = venv.get_solution()

    return sol


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    from args import argument_parser, parse_args

    parser = argument_parser()
    args, _ = parse_args(parser)

    assert (
        args.load_problem is not None
    ), "You should provide a problem to solve (use --load_problem)."

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
        unload=False,
    )
    agent = Agent.load(args.path, max_n_modes=problem_description.max_n_modes)

    agent.to(args.device)
    sol = solve_instance(problem_description, agent)
    sol.save(os.path.basename(psp.pb_id) + ".sol", psp.pb_id)
