import torch
import torch.nn as nn
from torch import Tensor 

# (B, T, input_dim)

# Upsampling by duplicating every element in the sequence
class Upsampling_EOE(nn.Module):
    def __init__(self, factor: int) -> None:
        super(Upsampling_EOE, self).__init__()
        self.factor = factor
    
    def forward(self, inputs: Tensor) -> torch.tensor:
        output = torch.repeat_interleave(inputs, self.factor, dim=1)
        return output 
    
# Downsampling by taking every other element (EOE)
class Downsampling_EOE(nn.Module):
    def __init__(self, factor: int) -> None:
        super(Downsampling_EOE, self).__init__()
        self.factor = factor
        
    def forward(self, inputs: Tensor) -> torch.tensor:
        outputs = inputs[:, ::self.factor, :]
        return outputs

# Upsampling with a linear projection (LP)
class Upsampling_LP(nn.Module):
    def __init__(self, input_dim: int, factor: int) -> None:
        super(Upsampling_LP, self).__init__()
        self.input_dim = input_dim
        self.factor = factor
        self.linear = nn.Linear(self.input_dim, self.input_dim*self.factor)
        
    def forward(self, inputs: Tensor) -> torch.tensor:
        outputs = self.linear(inputs)
        outputs = outputs.reshape(outputs.size(0), outputs.size(1) * self.factor , outputs.size(2) // self.factor )
        return outputs

# Downsampling with a linear projection (LP)
class Downsampling_LP(nn.Module):
    def __init__(self, input_dim: int, factor: int) -> None:
        super(Downsampling_LP, self).__init__()
        self.input_dim = input_dim
        self.factor = factor
        self.linear = nn.Linear(self.input_dim * self.factor, self.input_dim)
        
    def forward(self, inputs: Tensor) -> torch.tensor:
        outputs = inputs.reshape(inputs.size(0), inputs.size(1) // self.factor , self.input_dim * self.factor )
        outputs = self.linear(outputs)
        return outputs

# Downsampling with a mean operation
class Downsampling_Mean(nn.Module):
    def __init__(self, factor: int) -> None:
        super(Downsampling_Mean, self).__init__()
        self.factor = factor
        
    def forward(self, inputs: Tensor) -> torch.tensor:
        outputs = (inputs[:, :-1:self.factor] + inputs[:, 1::self.factor]) / 2
        return outputs