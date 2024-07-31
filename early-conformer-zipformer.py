import torch
from torch import nn
from torch import Tensor 
from torchaudio.models.conformer import Conformer

from utils.encoder import Encoder
from utils.positional_encoding import PositionalEncoding

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
    
class Conv1dSubamplingZip(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=0, padding_mode='zeros')
        
    def forward(self, inputs: Tensor) -> torch.tensor:
        outputs = self.conv(inputs)
        return outputs

class Early_conformer(nn.Module):
    
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
        self.pattern = [2,3,4,3,2]
        
        self.downsampling= nn.ModuleList([Downsampling(2**(factor-1)) for factor in self.pattern])
        self.downsampling_output = Downsampling(2)
        self.upsampling = nn.ModuleList([Upsampling(2**(factor-1)) for factor in self.pattern])
        self.conv_subsample = Conv1dSubamplingZip(in_channels=features_length, out_channels=d_model)
        self.positional_encoder = PositionalEncoding(d_model=d_model, dropout=drop_prob, max_len=max_len)
        self.linears=nn.Linear(d_model, dec_voc_size)
        self.conformer=nn.ModuleList([Conformer(input_dim=self.input_dim, num_heads=self.num_heads, ffn_dim=self.ffn_dim, num_layers=self.num_layers, depthwise_conv_kernel_size=self.depthwise_conv_kernel_size, dropout=self.dropout) for _ in range(self.n_enc_exits)])
    
    def forward(self, src, lengths):
        # enc = (B, T, input_dim)
        # n_enc_exits = 16 | n_enc_layers_per_exit = 1
        
        # Convolution 100Hz => 50Hz
        src = self.conv_subsample(src)
        src = self.positional_encoder(src.permute(0,2,1))
        length=torch.clamp(lengths/2,max=src.size(1)).to(torch.int).to(self.device)
        
        enc_out=[]
        base_length = length.to(self.device)
        enc = src
        
        enc, _ = self.conformer[0](enc, base_length)
        enc, _ = self.conformer[1](enc, base_length)
        
        for index in range(0, len(self.pattern)):
            
            src = enc
            
            factor = 2**(self.pattern[index]-1)
            conf_index = 2 + sum(self.pattern[:index])
            pad = enc.size(1) % factor
            if pad != 0:
                pad = factor - pad
                padding = torch.zeros(enc.size(0), pad, enc.size(2), device=self.device)
                enc = torch.cat((enc, padding), dim=1)
                
            enc = self.downsampling[index](enc)
            length = torch.clamp((lengths + pad)/factor, max=enc.size(1)).to(torch.int).to(self.device)
            
            for i in range(conf_index, conf_index + self.pattern[index]):   
                enc, _ = self.conformer[i](enc, length)
            
            enc = self.upsampling[index](enc)
            
            if pad != 0:
                enc = enc[:, :-pad, :]
            length = torch.clamp(base_length, max=enc.size(1)).to(torch.int).to(self.device)
            
            enc = enc + src
            
        # 50Hz => 25Hz
        out = self.downsampling_output(enc)
        out = self.linears(out)
        out = torch.nn.functional.log_softmax(out,dim=2)
        enc_out += [out.unsqueeze(0)]
        enc_out = torch.cat(enc_out)
        
        return enc_out 