import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


##########
##  ReAct
##########

class LearnableBias(nn.Module):
    def __init__(self, in_channels):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(
                        torch.zeros(1,in_channels,1,1), 
                        requires_grad=True
                    )

    def forward(self, input):
        return input + self.bias.expand_as(input)

class RPReLU(nn.Module):
    def __init__(self, in_channels):
        super(RPReLU, self).__init__()
        self.shift_x = LearnableBias(in_channels)
        self.shift_y = LearnableBias(in_channels)
        self.prelu = nn.PReLU(in_channels)

    def forward(self, input):
        input = self.shift_x(input)
        input = self.prelu(input)
        input = self.shift_y(input)
        return input

class RSign(nn.Module):
    def __init__(self, in_channels):
        super(RSign, self).__init__()
        self.shift_x = LearnableBias(in_channels)
        self.binarize = FastSign()

    def forward(self, input):
        input = self.shift_x(input)
        input = self.binarize(input)
        return input


##########
##  Quant
##########

class FastSign(nn.Module):
    def __init__(self):
        super(FastSign, self).__init__()

    def forward(self, input):
        out_forward = torch.sign(input)
        ''' 
        Only inputs in the range [-t_clip,t_clip] 
        have gradient 1. 
        '''
        t_clip = 1.3
        out_backward = torch.clamp(input, -t_clip, t_clip)
        return (out_forward.detach() 
                - out_backward.detach() + out_backward)

class BirealBinarize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input = grad_input * (input > -1.0).float()
        grad_input = grad_input * (input < 1.0).float()
        mask = (input < 0.0).float()
        grad_input = grad_input * ( 
                        ((2.0+2.0*input) * mask 
                        + (2.0-2.0*input) * (1.0-mask)) 
                     )
        return grad_input

class BirealSignActivation(nn.Module):
    '''
    This module, together with BirealBinarize, implements 
    a memory efficient version of the activation function 
    used in Bi-real Net and ReActNet.
    Reference:
        https://github.com/liuzechun/Bi-Real-net 
    '''
    def __init__(self):
        super(BirealSignActivation, self).__init__()

    def forward(self, input):
        return BirealBinarize.apply(input)

class QuantSign(torch.autograd.Function):
    '''
    Quantize Sign activation to arbitrary bitwidth.
    Usage: 
        output = QuantSign.apply(input, bits)
    '''
    @staticmethod
    def forward(ctx, input, bits=2):
        ctx.save_for_backward(input)
        input = torch.clamp(input, -1.0, 1.0)
        delta = 2.0/(2.0**bits-1.0)
        input = torch.round((input+1.0)/delta)*delta-1.0
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        ''' 
        Only inputs in the range [-t_clip,t_clip] 
        have gradient 1. 
        '''
        t_clip = 1.0
        grad_input = grad_output.clone()
        grad_input *= (input>-t_clip).float()
        grad_input *= (input<t_clip).float()
        return grad_input, None

class FastTruncate(nn.Module):
    ''' Truncate the MSB of a 2-bit value '''
    def __init__(self):
        super(FastTruncate, self).__init__()

    def forward(self, input):
        ''' Subtract 1e-6 for floor() '''
        out_forward = torch.clamp(input, -1.0, 1.0-1e-6)
        ''' Quantize values to either 0*delta or 2*delta '''
        out_forward = torch.floor(out_forward+1.0)*2.0
        delta = 2.0 / (2.0**2-1.0)
        out_forward = out_forward*delta-1.0
        ''' 
        Only inputs in the range [-t_clip,t_clip] 
        have gradient 1. 
        '''
        t_clip = 1.0
        out_backward = torch.clamp(input, -t_clip, t_clip)
        return (out_forward.detach() 
                - out_backward.detach() + out_backward)


##########
##  Mask
##########

class SparseGreaterThan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        ctx.save_for_backward(input, torch.tensor(threshold))
        return torch.Tensor.float(torch.gt(input, threshold))

    @staticmethod
    def backward(ctx, grad_output):
        input, threshold, = ctx.saved_tensors
        grad_input = grad_output.clone()
        ''' Identity gradients only when input >= threshold '''
        grad_input *= (input>=threshold).float()
        return grad_input, None

class GreaterThan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        return torch.Tensor.float(torch.gt(input, threshold))

    @staticmethod
    def backward(ctx, grad_output):
        ''' Identity gradients '''
        grad_input = grad_output.clone()
        return grad_input, None


##########
##  Layer
##########

class BinaryConv2d(nn.Conv2d):
    '''
    A convolutional layer with its weight tensor binarized to {-1, +1}.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros'):
        super(BinaryConv2d, self).__init__(in_channels, out_channels,
                                              kernel_size, stride,
                                              padding, dilation, groups,
                                              bias, padding_mode)
        self.binarize = FastSign()
        # self.binarize = nn.Sequential()

    def forward(self, input):
        return F.conv2d(input, self.binarize(self.weight),
                        self.bias, self.stride, self.padding,
                        self.dilation, self.groups)

class PGBinaryConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False,
                 padding_mode='zeros', sparse_bp=True, init=-1.0):
        super(PGBinaryConv2d, self).__init__(in_channels, out_channels,
                                       kernel_size, stride,
                                       padding, dilation, groups,
                                       bias, padding_mode)
        self.binarize = FastSign()
        # self.binarize = nn.Sequential()
        self.truncate_a = FastTruncate()
        self.gt = SparseGreaterThan.apply if sparse_bp else GreaterThan.apply

        '''
        zero initialization
        nan loss while using torch.Tensor to initialize the thresholds
        '''
        self.threshold = nn.Parameter(torch.ones(1, out_channels, 1, 1)*init)

        ''' number of output features '''
        self.num_out = 0
        ''' number of output features computed at high precision '''
        self.num_high = 0

    def forward(self, input):
        ''' MSB convolution '''
        out_msb = F.conv2d(self.truncate_a(input),
                           self.binarize(self.weight),
                           self.bias, self.stride, self.padding,
                           self.dilation, self.groups)
        ''' Calculate the mask '''
        # TODO: check whether the .detach() here decrease the accuracy
        #mask = self.gt(torch.sigmoid(5.0*(out_msb-self.threshold)), 0.5)
        mask = self.gt(
                F.hardtanh(
                    0.25*(out_msb-self.threshold+2.0), 
                    min_val=0.0, 
                    max_val=1.0
                ), 
                0.5
               )
        ''' update report '''
        self.num_out = mask.numel()
        self.num_high = (mask>0).sum().item()
        ''' full convolution '''
        out_full = F.conv2d(input,
                           self.binarize(self.weight),
                           self.bias, self.stride, self.padding,
                           self.dilation, self.groups)
        ''' combine outputs '''
        return (1-mask)*out_msb + mask * out_full


##########
##  Transform
##########

class InputEncoder(nn.Module):
    '''
    Encode the input images to bipolar strings using thermometer encoding.
    Request:
        Know the input size beforehand.
    '''
    def __init__(self, input_size, resolution):
        super(InputEncoder, self).__init__()
        self.n, self.c, self.h, self.w = input_size
        self.resolution = int(resolution)
        self.b = int(round(255.0/self.resolution))
        placeholder = torch.ones(self.n, self.c, self.b, self.h, self.w, \
                                      dtype=torch.float32).cuda() if \
                           torch.cuda.is_available() else \
                           torch.ones(self.n, self.c, self.b, self.h, self.w, \
                                      dtype=torch.float32)
        placeholder *= torch.arange(self.b).view(1,1,-1,1,1).cuda() if \
                            torch.cuda.is_available() else \
                            torch.arange(self.b).view(1,1,-1,1,1)
        self.register_buffer('placeholder', placeholder)

    def forward(self, x):
        x = (x * 255.0).view(-1, self.c, 1, self.h, self.w)
        mask = (self.placeholder < torch.round(x/self.resolution)).float()
        output = (1.0*mask + (-1.0)*(1-mask))
        return output.view(-1, self.b*self.c, self.h, self.w).detach()




##########
##  Cached
##########

"""
class RPReLU(nn.Module):
    '''
    ReAct PReLU
    Moving the origin of PReLU from (0,0) to (gamma, yita).
    Each channel has its own parameters beta, gamma, and yita.
    beta: slope for input<0.
    '''
    def __init__(self, in_channels=1, init_beta=0.25,
                 init_gamma=0., init_yita=0.):
        super(RPReLU, self).__init__()
        self.beta = nn.Parameter(torch.ones(1,in_channels,1,1)*init_beta)
        self.gamma = nn.Parameter(torch.ones(1,in_channels,1,1)*init_gamma)
        self.yita = nn.Parameter(torch.ones(1,in_channels,1,1)*init_yita)

    def forward(self, input):
        mask1 = (input > self.gamma).float()
        mask2 = 1.0 - mask1
        output = (input-self.gamma+self.yita)*mask1 + \
                 (self.beta*(input-self.gamma)+self.yita)*mask2
        return output
"""

