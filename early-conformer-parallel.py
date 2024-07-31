import torch
from torch import nn
from torch import Tensor 
from torchaudio.models.conformer import Conformer

from utils.encoder import Encoder
from utils.positional_encoding import PositionalEncoding

class Conv1dSubampling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(Conv1dSubampling, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=0, padding_mode='zeros'),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=2, padding=0, padding_mode='zeros')
        )
    def forward(self, inputs: Tensor) -> torch.tensor:
        outputs = self.sequential(inputs)
        return outputs 

class Upsampling(nn.Module):
    def __init__(self, factor: int) -> None:
        super(Upsampling, self).__init__()
        self.factor = factor
    
    def forward(self, inputs: Tensor) -> torch.tensor:
        # (B, T, input_dim)
        output = torch.repeat_interleave(inputs, self.factor, dim=1)
        return output 
    
class Downsampling(nn.Module):
    def __init__(self, factor: int) -> None:
        super(Downsampling, self).__init__()
        self.factor = factor
        
    def forward(self, inputs: Tensor) -> torch.tensor:
        outputs = inputs[:, ::self.factor, :]
        return outputs

class Early_conformer(nn.Module):
    # Working here
    def __init__(self, src_pad_idx, n_enc_exits, enc_voc_size, dec_voc_size, d_model, n_head, max_len,  d_feed_forward, n_enc_layers,  features_length, drop_prob, depthwise_kernel_size, device):
        super().__init__()
        self.input_dim=d_model
        self.num_heads=n_head
        self.ffn_dim=d_feed_forward
        self.num_layers=n_enc_layers
        self.depthwise_conv_kernel_size=depthwise_kernel_size
        self.n_enc_exits=n_enc_exits
        self.dropout=drop_prob
        self.device=device
        self.src_pad_idx=src_pad_idx
        
        self.downsampling= nn.ModuleList([Downsampling(2) for _ in range(self.n_enc_exits//3)])
        self.upsampling = nn.ModuleList([Upsampling(2) for _ in range(self.n_enc_exits//3)])
        self.conv_subsample = Conv1dSubampling(in_channels=features_length, out_channels=d_model)
        self.positional_encoder = PositionalEncoding(d_model=d_model, dropout=drop_prob, max_len=max_len)
        self.linears=nn.Linear(d_model, dec_voc_size)
        self.conformer=nn.ModuleList([Conformer(input_dim=self.input_dim, num_heads=self.num_heads, ffn_dim=self.ffn_dim, num_layers=self.num_layers, depthwise_conv_kernel_size=self.depthwise_conv_kernel_size, dropout=self.dropout) for _ in range(self.n_enc_exits)])
        
    def forward(self, src, lengths):
        # enc = (B, T, input_dim)
        # n_enc_exits = 24 | n_enc_layers = 1 => 1 exit
        
        src = self.conv_subsample(src)
        src = self.positional_encoder(src.permute(0,2,1))
        length=torch.clamp(lengths/4,max=src.size(1)).to(torch.int).to(self.device)  
        
        enc_out= []
        base_length = length.to(self.device)
        pad = 0
        factor = 2
        enc_normal = src
        
        for index in range(0, self.n_enc_exits//3):
            
            enc_downsampled = enc_normal 
            
            enc_normal, _ = self.conformer[3*index](enc_normal, base_length)
            enc_normal, _ = self.conformer[3*index+1](enc_normal, base_length)
            
            pad = enc_downsampled.size(1) % factor
    
            if pad != 0:
                    pad = factor - pad
                    padding = torch.zeros(enc_downsampled.size(0), pad, enc_downsampled.size(2), device=self.device)
                    enc_downsampled = torch.cat((enc_downsampled, padding), dim=1)
            enc_downsampled = self.downsampling[index](enc_downsampled)
            length = torch.clamp((lengths + pad)/factor, max=enc_downsampled.size(1)).to(torch.int).to(self.device)
             
            enc_downsampled, _ = self.conformer[3*index + 2](enc_downsampled, length)
            
            enc_downsampled = self.upsampling[index](enc_downsampled)
            if pad != 0:
                enc_downsampled = enc_downsampled[:, :-pad, :]
            
            length = torch.clamp(base_length, max=enc_downsampled.size(1)).to(torch.int).to(self.device)
            
            enc_normal = enc_downsampled + enc_normal
            
        out = self.linears(enc_normal)
        out = torch.nn.functional.log_softmax(out,dim=2)
        enc_out += [out.unsqueeze(0)]
        enc_out = torch.cat(enc_out)
        
        return enc_out 