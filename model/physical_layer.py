import torch
import torch.nn as nn
import numpy as np
import os
from utils import discretize,findzero


class quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx,input):
        return torch.round(input*255) / 255
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = torch.clamp(grad_output, -1, 1)
        return grad_input
quantizer = quantize.apply


class encoder(nn.Module):
    def __init__(self, args):
        super(encoder,self).__init__()
        self.gray_levels = args.gray_levels
        self.size = args.size
        self.B = args.B
        self.train_type = args.train_type
        if self.train_type == 'sim':
            if self.gray_levels == 2:
                self.weight = nn.Parameter(torch.Tensor(self.size[0], self.size[1]))    # [H,W] 
                self.index = torch.arange(0, args.B)[:, None, None].repeat(1, args.size[0], args.size[1])
            else:
                self.weight = nn.Parameter(torch.Tensor(self.B, self.size[0], self.size[1]))    # [B,H,W]
        elif self.train_type == 'real':
            self.weight = nn.Parameter(torch.Tensor(self.B, int(self.size[0] / 2), int(self.size[1] / 2)))  # [B,H/2,W/2]
        else:
            raise NameError('Please determine the type of encoder train, [sim] or [real].')
        self.reset_parameters()


    def reset_parameters(self):
        self.weight.detach().uniform_(0, 1.0)  
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.detach().clone().float()
        self.weight.data = discretize(self.weight.org, self.B, self.gray_levels)

    def generate_mask(self):
        if self.train_type == 'sim':
            if self.gray_levels == 2:
                mask = findzero(self.index.to(self.weight) - self.weight).float()    
            else:
                mask = self.weight.float()   # [B,H,W]
        elif self.train_type == 'real':
            mask = torch.repeat_interleave(torch.repeat_interleave(self.weight,2, dim=1), 2, dim=2).float()
        else:
            raise NameError('Please determine the type of encoder train, [sim] or [real].')
        return mask 

    def forward(self, gt_batch):
        mask = self.generate_mask()
        mask_batch = mask.repeat((gt_batch.shape[0],1,1,1))
        meas_batch = torch.mul(mask_batch, gt_batch)
        meas_batch = torch.sum(meas_batch, dim=1)  # [batch,256 256]    
        meas_batch = torch.clamp(meas_batch, 0, 1) # dynamic range clipping
        meas_batch = quantizer(meas_batch)         # quantization [0, 1/255, 2/255,..., 1]
        meas_batch = meas_batch.float() 

        return meas_batch, mask