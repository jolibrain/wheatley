from utils.utils import get_n_features


class EnvSpecification:
    def __init__(
        self,
        max_n_jobs,
        max_n_machines,
        normalize_input,
        input_list,
        insertion_mode,
        max_edges_factor,
        sample_n_jobs,
        chunk_n_jobs,
    ):
        self.max_n_jobs = max_n_jobs
        self.max_n_machines = max_n_machines
        self.max_n_nodes = max_n_jobs * max_n_machines
        self.max_n_edges = self.max_n_nodes ** 2
        self.normalize_input = normalize_input
        self.input_list = input_list
        self.insertion_mode = insertion_mode
        self.max_edges_factor = max_edges_factor
        self.sample_n_jobs = sample_n_jobs
        self.chunk_n_jobs = chunk_n_jobs
        self.add_boolean = (insertion_mode == "choose_forced_insertion") or (insertion_mode == "slot_locking")
        self.n_features = get_n_features(self.input_list, self.max_n_jobs, self.max_n_machines)

    def print_self(self):
        print_input_list = [el.lower().title().replace("_", " ") for el in self.input_list]
        print(
            f"==========Env Description     ==========\n"
            f"Max size:                         {self.max_n_jobs} x {self.max_n_machines}\n"
            f"Input normalization:              {'Yes' if self.normalize_input else 'No'}\n"
            f"Insertion mode:                   {self.insertion_mode.lower().title().replace('_', ' ')}\n"
            f"List of features:"
        )
        print(" - " + "\n - ".join(print_input_list) + "\n")
