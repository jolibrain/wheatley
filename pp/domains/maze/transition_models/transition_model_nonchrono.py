from ..state import InvalidSelectionException
import torch


class MazeTransitionModel:
    def __init__(self, env, nonchrono):
        self.env = env
        self.nonchrono = nonchrono  # "wp", "wpr" or "path"

    def get_mask(self, state):
        mask = state.get_mask()
        # print('mask sum=0: ', mask.sum() == 0)
        return mask

    def run(self, state, nid):
        if self.nonchrono == "wp":
            if not state.selected[nid]:
                state.select(nid)
            else:
                print(f"invalid selection of {nid}")
                raise InvalidSelectionException(nid)
        elif self.nonchrono == "wpr":
            state.select_wpr(nid)
        else:
            state.update_path(nid)

        return state
