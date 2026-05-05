import numpy as np

from maze_dataset.maze import SolvedMaze, LatticeMaze
from muutils.misc import stable_hash

class PathPlanningProblem:
    def __init__(self, solvedMaze):
        self.solvedMaze = solvedMaze
        self.x, self.y = self.solvedMaze.grid_shape
        self.shape = self.solvedMaze.grid_shape
        self.ncells = self.x * self.y
        self.optimal = np.array(solvedMaze.solution, copy=True)
        self._connection_list = np.array(self.solvedMaze.connection_list, copy=True)
        self._initial_connection_list = np.array(
            self._connection_list,
            copy=True,
        )
        self._generation_meta = self.solvedMaze.generation_meta
        self._starts_cache = {}
        self._start_seed = abs(hash(self.data_hash())) % (2**32)
        self._goal_cache = {}
        self._goal_seed = (self._start_seed + 1) % (2**32)
        self._danger_cache = {}
        self._danger_seed = (self._start_seed + 2) % (2**32)

    def ncells(self):
        return self.ncells

    def start(self):
        return self.solvedMaze.start_pos

    def agent_starts(self, num_agents, force_common_start=False):
        """Return a list of start coordinates, one per agent."""
        if num_agents <= 0:
            return []
        if num_agents == 1 or force_common_start:
            return [tuple(self.start())] * num_agents

        cache_key = (num_agents, bool(force_common_start))
        if cache_key in self._starts_cache:
            return self._starts_cache[cache_key]

        base_start = tuple(self.start())
        goal = tuple(self.end())
        candidates = [
            tuple(c)
            for c in self.nodes_coord()
            if tuple(c) not in (base_start, goal)
        ]
        rng = np.random.default_rng(self._start_seed)
        rng.shuffle(candidates)

        starts = [base_start]
        for cand in candidates:
            if len(starts) == num_agents:
                break
            starts.append(cand)

        if len(starts) < num_agents and candidates:
            idx = 0
            while len(starts) < num_agents:
                starts.append(candidates[idx % len(candidates)])
                idx += 1

        if len(starts) < num_agents:
            starts = starts + [base_start] * (num_agents - len(starts))

        self._starts_cache[cache_key] = starts
        return starts

    def shared_goals(self, num_goals, exclude_coords=None):
        """Return a list of shared goal coordinates."""
        if num_goals <= 0:
            return []

        cache_key = (num_goals, tuple(sorted(set(exclude_coords or []))))
        if cache_key in self._goal_cache:
            return self._goal_cache[cache_key]

        goals = []
        exclude = set(tuple(c) for c in (exclude_coords or []))
        exclude.add(tuple(self.start()))

        primary_goal = tuple(self.end())
        if primary_goal not in exclude:
            goals.append(primary_goal)
            exclude.add(primary_goal)

        rng = np.random.default_rng(self._goal_seed)
        candidates = [
            tuple(c)
            for c in self.nodes_coord()
            if tuple(c) not in exclude and tuple(c) != tuple(self.start())
        ]
        rng.shuffle(candidates)

        for cand in candidates:
            if len(goals) >= num_goals:
                break
            if cand in exclude:
                continue
            goals.append(cand)
            exclude.add(cand)

        if len(goals) < num_goals:
            all_nodes = [
                tuple(c)
                for c in self.nodes_coord()
                if tuple(c) not in exclude
            ]
            for cand in all_nodes:
                if len(goals) >= num_goals:
                    break
                goals.append(cand)
                exclude.add(cand)

        if not goals:
            goals.append(primary_goal)

        if len(goals) < num_goals:
            raise ValueError(
                f"Unable to sample {num_goals} unique goals from maze with {len(goals)} candidates"
            )
        goals = goals[:num_goals]
        self._goal_cache[cache_key] = goals
        return goals

    def danger_zones(self, max_num, max_size):
        """Return a list of rectangular danger zones as coordinate lists."""
        if max_num <= 0 or max_size <= 0:
            return []

        max_num = int(max_num)
        max_size = int(max_size)
        cache_key = (max_num, max_size)
        if cache_key in self._danger_cache:
            return self._danger_cache[cache_key]

        seed_offset = stable_hash(str(cache_key))
        rng = np.random.default_rng((self._danger_seed + seed_offset) % (2**32))
        max_height = max(1, min(max_size, self.x))
        max_width = max(1, min(max_size, self.y))
        num_zones = int(rng.integers(0, max_num + 1))
        zones = []
        for _ in range(num_zones):
            height = int(rng.integers(1, max_height + 1))
            width = int(rng.integers(1, max_width + 1))
            max_row = max(0, self.x - height)
            max_col = max(0, self.y - width)
            row = int(rng.integers(0, max_row + 1)) if max_row >= 0 else 0
            col = int(rng.integers(0, max_col + 1)) if max_col >= 0 else 0
            coords = [
                (row + dr, col + dc)
                for dr in range(height)
                for dc in range(width)
            ]
            zones.append(coords)

        self._danger_cache[cache_key] = zones
        return zones

    def end(self):
        return self.solvedMaze.end_pos

    def nodes_coord(self):
        return self.solvedMaze.get_nodes()

    def adj_list(self):
        return self.solvedMaze.as_adj_list()

    def neigh_of(self, coord):
        return self.solvedMaze.get_coord_neighbors(coord)

    def data_hash(self):
        return stable_hash(str(self.solvedMaze.serialize()))

    def break_wall(self, coord_a, coord_b):
        """Open a wall between two adjacent coordinates in the maze dataset view."""
        dim, x, y = self._connection_index(coord_a, coord_b)
        if self._connection_list[dim, x, y]:
            return False
        self._connection_list[dim, x, y] = True
        self._refresh_solved_maze()
        return True

    def reset_dynamic_changes(self):
        """Restore the maze to its original connectivity."""
        self._connection_list = np.array(
            self._initial_connection_list,
            copy=True,
        )
        self._refresh_solved_maze()

    def _connection_index(self, coord_a, coord_b):
        a = np.array(coord_a, dtype=int)
        b = np.array(coord_b, dtype=int)
        delta = b - a
        if np.abs(delta).sum() != 1:
            raise ValueError(
                f"coords must be adjacent to break wall, got {coord_a} and {coord_b}"
            )
        dim = int(np.argmax(np.abs(delta)))
        clist_node = a if delta.sum() > 0 else b
        return dim, int(clist_node[0]), int(clist_node[1])

    def _refresh_solved_maze(self):
        self.solvedMaze = SolvedMaze(
            connection_list=np.array(self._connection_list, copy=True),
            solution=np.array(self.optimal, copy=True),
            generation_meta=self._generation_meta,
        )
