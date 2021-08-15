import torch
import torch.nn as nn

class DenseNormalGamma(nn.Module):
    def __init__(self, units_in, units_out):
        super(DenseNormalGamma, self).__init__()
        self.units_in = int(units_in)
        self.units_out = int(units_out)
        self.linear = nn.Linear(units_in, 4 * units_out)

    def evidence(self, x):
        softplus = nn.Softplus(beta=1)
        return softplus(x)

    def forward(self, x):
        output = self.linear(x)
        mu, logv, logalpha, logbeta = torch.split(output, self.units_out, dim=-1)
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)
        return torch.cat(tensors=(mu, v, alpha, beta), dim=-1)

    def compute_output_shape(self):
        return (self.units_in, 4 * self.units_out)
