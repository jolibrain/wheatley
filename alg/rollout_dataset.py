import torch


def collate_rollout(data, agent=None):
    obs, logprobs, actions, advantages, returns, values, actions_masks = zip(*data)

    return (
        agent.preprocess(obs),
        torch.stack(logprobs),
        torch.stack(actions),
        torch.stack(advantages),
        torch.stack(returns),
        torch.stack(values),
        torch.stack(actions_masks),
    )


class RolloutDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        agent,
        obs,
        logprobs,
        actions,
        advantages,
        returns,
        values,
        actions_masks,
        sigma,
    ):
        self.agent = agent
        self.obs = obs
        self.logprobs = logprobs
        self.actions = actions
        self.advantages = advantages
        self.returns = returns
        self.values = values
        self.actions_masks = actions_masks
        self.sigma = sigma

    def __len__(self):
        return self.logprobs.shape[0]

    def __getitem__(self, idx):
        return (
            self.agent.get_obs(self.obs, [idx])[0],
            self.logprobs[idx],
            self.actions[idx],
            self.advantages[idx],
            self.returns[idx],
            self.values[idx],
            self.actions_masks[idx],
        )
