class TRandomFieldCell(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def set_mean(self, mean):
        self.mean = mean

    def set_std(self, std):
        self.std = std

    def get_mean(self):
        return self.mean

    def get_std(self):
        return self.std