#
# Wheatley
# Copyright (c) 2023 Jolibrain
# Authors:
#    Guillaume Infantes <guillaume.infantes@jolibrain.com>
#    Antoine Jacquet <antoine.jacquet@jolibrain.com>
#    Michel Thomazo <thomazo.michel@gmail.com>
#    Emmanuel Benazera <emmanuel.benazera@jolibrain.com>
#
#
# This file is part of Wheatley.
#
# Wheatley is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Wheatley is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Wheatley. If not, see <https://www.gnu.org/licenses/>.
#
class Description:
    def __init__(
        self,
        transition_model_config,
        reward_model_config,
        train_pp,
        test_pp,
        seed,
        unload=False,
    ):
        """ """

        self.transition_model_config = transition_model_config
        self.reward_model_config = reward_model_config
        self.train_pbs = train_pp
        self.test_pbs = test_pp if len(test_pp) != 0 else train_pp
        self.seed = seed
        self.unload = unload

        if self.unload:
            self.train_pbs_ids = [pp.pb_id for pp in train_pp]
            self.train_pbs = None
            self.test_pbs_ids = [psp.pb_id for psp in test_pp]
            self.test_pbs = None
            self.ntrain = len(self.train_pbs_ids)
            self.ntest = len(self.test_pbs_ids)
        else:
            self.ntrain = len(self.train_pbs)
            self.ntest = len(self.test_pbs)

    def print_self(self):
        print(
            f"==========Problem Description ==========\n"
            f"Transition model:                 {self.transition_model_config}\n"
            f"Reward model:                     {self.reward_model_config}\n"
            f"Train set size:                   {self.ntrain}\n"
            f"Test set size:                    {self.ntest}\n"
        )
