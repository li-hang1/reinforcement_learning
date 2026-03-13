from torch.distributions import Normal
import torch


mu = torch.tensor(2.0, requires_grad=True)
sigma = torch.tensor(1.0)
dist = Normal(mu, sigma)
a = dist.sample()
b = dist.rsample()
print(a)
print(b)
print(dist.log_prob(a))