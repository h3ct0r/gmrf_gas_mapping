class TObservationGMRF(object):
    def __init__(self, obs_value, obs_lambda, time_invariant):
        self.obs_value = obs_value
        self.obs_lambda = obs_lambda
        self.time_invariant = time_invariant

    def set_obs_value(self, obs_value):
        self.obs_value = obs_value

    def set_obs_lambda(self, obs_lambda):
        self.obs_lambda = obs_lambda

    def set_time_invariant(self, time_invariant):
        self.time_invariant = time_invariant

    def get_obs_value(self):
        return self.obs_value

    def get_obs_lambda(self):
        return self.obs_lambda

    def get_time_invariant(self):
        return self.time_invariant