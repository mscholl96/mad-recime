import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.batchnorm import BatchNorm1d
from torch.nn.modules.linear import Linear

class ReLUBatchNormLayer(nn.Module):
    def __init__(self, input_number, output_number):
        super(ReLUBatchNormLayer, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(input_number, output_number),
            nn.BatchNorm1d(output_number),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x)

class VariationalAutoEncoder(nn.Module):
    def __init__(self,D_in,H=1024,H2=128,latent_dim=32):
        super(VariationalAutoEncoder,self).__init__()

        # Encoder stack
        # From input all the way to the latent space
        self.encoderStack = nn.Sequential(
            ReLUBatchNormLayer(D_in, H),
            ReLUBatchNormLayer(H, H2),
            ReLUBatchNormLayer(H2, H2),
            ReLUBatchNormLayer(H2, latent_dim),
        )
        # Layers for mu and sigma (Belong to encoder stack)
        self.mu = nn.Linear(latent_dim, latent_dim)
        self.logvar = nn.Linear(latent_dim, latent_dim)

        # Decoder stack
        self.decoderStack = nn.Sequential(
            ReLUBatchNormLayer(latent_dim, latent_dim),
            ReLUBatchNormLayer(latent_dim, H2),
            ReLUBatchNormLayer(H2, H2),
            ReLUBatchNormLayer(H2, H),
            nn.Linear(H, D_in),
            nn.BatchNorm1d(D_in),
        )
        
    def encode(self, x):
        latent_space = self.encoderStack(x)

        mu = self.mu(latent_space)
        logvar = self.logvar(latent_space)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu
        
    def decode(self, z):
        return self.decoderStack(z)
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar