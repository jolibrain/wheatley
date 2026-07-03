from ..state import InvalidMoveException
import torch


class MazeTransitionModel:
    def __init__(self, env):
        self.env = env

    def allow_neighbor1_nodes(self, state):
        return self.env.pos_neigh_nodes(state.pos)

    def allow_all_neithbor1_nodes(self, state):
        return self.env.all_pos_neigh_nodes(state.pos)

    def unvisited_neigh_nodes(self, state):
        neigh_nodes = torch.tensor([False] * state.n_nodes)
        neigh_nodes[self.env.pos_neigh_nodes(state.pos)] = True
        neigh_and_unvisited = torch.logical_and(neigh_nodes, torch.eq(state.visited, 0))
        return torch.where(neigh_and_unvisited)[0]

    def get_mask(self, state):
        if state.n_breaks > 0:
            allowed_nodes = self.allow_all_neithbor1_nodes(state)
        else:
            allowed_nodes = self.allow_neighbor1_nodes(state)
        mask = torch.zeros(state.n_nodes, dtype=torch.bool)

        mask[allowed_nodes] = True  # True means allowed atm

        # mask = torch.logical_and(mask, state.visited == 0)
        if getattr(self.env, "allow_stay", False):
            # allow noop by selecting the current position
            mask[state.pos] = True
        return mask

    def run(self, state, nid):
        state.clear_goal_contact()
        if getattr(self.env, "allow_stay", False) and nid == state.pos:
            # No-op move: counted as a step for the agent/environment
            state.move_to(nid)
        else:
            if state.n_breaks > 0:
                if nid in self.env.all_pos_neigh_nodes(state.pos):
                    state.move_to(nid)
                else:
                    print(f"invalid move from {state.pos} to {nid}")
                    raise InvalidMoveException(state.pos, nid)
            else:
                if nid in self.env.pos_neigh_nodes(state.pos):
                    state.move_to(nid)
                else:
                    print(f"invalid move from {state.pos} to {nid}")
                    raise InvalidMoveException(state.pos, nid)

        consumed_idx = self.env.try_consume_goal(state.pos)
        if consumed_idx is not None:
            state.on_goal_consumed(consumed_idx)

        return state
