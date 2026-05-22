class Solution:
    def __init__(self, path, reward_model, optimal, path_native, failed):
        self.sol = path
        if path is None or failed:
            self.criterion = reward_model.fail_cost
        else:
            self.criterion = reward_model.unit_cost * len(path)
        self.optimal = optimal
        self.optimal_criterion = reward_model.unit_cost * (len(optimal) - 1)
        self.path_native = path_native
        self.failed = failed

    def get_criterion(self):
        return self.criterion

    def get_optimal_criterion(self):
        return self.optimal_criterion

    def get_native_sol(self):
        return self.path_native

    def save(self, path):
        with open(path, "w") as f:
            f.write("agent: " + str(self.sol))
            f.write("\noptimal: " + str(self.optimal))
