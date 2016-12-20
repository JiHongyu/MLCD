class Algorithm:

    def __init__(self, networks):
        self.networks = networks
        self.r = {'node_coms':[], 'link_coms':[]}

    def run_algo(self):
        return self.r

    @property
    def result(self):
        self.r