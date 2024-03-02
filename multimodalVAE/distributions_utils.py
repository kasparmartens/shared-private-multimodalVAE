from torch.distributions.poisson import Poisson
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.categorical import Categorical

class myPoisson(Poisson):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def log_prob(self, x):
        return super().log_prob(x.int())

class myCategorical(Categorical):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def log_prob(self, x):
        return super().log_prob(x.reshape(-1)).reshape(-1, 1)
