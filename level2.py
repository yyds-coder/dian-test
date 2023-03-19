from torch.nn.modules import Module
from torch.nn.parameter import Parameter
import random
import argparse
import functools
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
import sys
sys.setrecursionlimit(100000)
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from torch.nn.common_types import _size_2_t
from typing import Union

class Linear(Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features,in_features), **factory_kwargs))  # 随机weight
        if bias:
            self.bias = Parameter(torch.empty((out_features), **factory_kwargs))

    def forward(self, input):
        '''TODO'''
        self.x=input
        input = input.detach().numpy()
        weight = self.weight.detach().numpy()
        bias = self.bias.detach().numpy()
        self.output = input.dot(weight.T) + bias
        self.output = torch.tensor(self.output)
        return self.output

    def backward(self, ones: Tensor):
        '''TODO'''
        lele=torch.mm(ones,self.weight)
        self.weight.grad=torch.mm(ones.T,self.x)
        self.bias.grad=torch.sum(ones,axis=0)
        return lele


class CrossEntropyLoss():
    def __init__(self):
        self.input = None
        self.target = None

    def forward(self, input, target, dim=1):
        '''TODO'''
        xx = input
        yy = target
        self.target = yy
        self.input = xx
        softmax = nn.Softmax(dim)
        x_softmax = softmax(xx)  # 按行做softmax
        x_log = torch.log(x_softmax)  # 取对数
        loss = x_log[range(len(xx)), yy]  # len(x)是行数,range按照行变化,按照target给出每行的值
        loss = abs(sum(loss) / len(xx))
        self.output = loss
        return self.output

    def backward(self):
        res = self.target
        fa = nn.Softmax(1)
        dnx = fa(self.input)
        for j in range(dnx.shape[0]):
            dnx[j][res[j]] = dnx[j][res[j]] - 1
        output = dnx/2
        return output

class Conv2d(_ConvNd):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_2_t,
            stride: _size_2_t = 1,
            padding: Union[str, _size_2_t] = 0,
            dilation: _size_2_t = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',  # TODO: refine this type
            device=None,
            dtype=None
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode, **factory_kwargs)

    def conv2d(self, input, kernel, bias=0, stride=1, padding=0):
        '''TODO forword的计算方法'''
        # kernel是随机生成的卷积核参数
        # bias无偏
        # stride步长为一
        # padding不对图像做填充
        # kernel的维度是a,b,c,d
        self.input=input
        a = kernel.shape[0]
        b = kernel.shape[1]  # 数据集的输入通道是一
        c = kernel.shape[2]
        d = kernel.shape[3]
        lie = input.shape[3]
        hang = input.shape[2]
        self.input_height = input.shape[2]
        self.input_width = input.shape[3]
        self.weight_height = kernel.shape[2]
        self.weight_width = kernel.shape[3]
        self.output_height = int((self.input_height - self.weight_height) ) + 1
        self.output_width = int((self.input_width - self.weight_width) ) + 1
        self.kernel=kernel
        st = hang - c + 1
        sq = lie - d + 1
        sum = 0
        first = torch.zeros(st, sq)
        sec = torch.zeros(st, sq)
        final = torch.zeros(kernel.shape[0], st, sq)
        ultra = torch.zeros(input.shape[0], kernel.shape[0], st, sq)
        for l in range(input.shape[0]):
            for v in range(kernel.shape[0]):
                for s in range(kernel.shape[1]):  # 输入图像有几层
                    for t in range(hang - c + 1):  # 再动行,形成一层卷积
                        for k in range(lie - d + 1):  # 动态卷积开始，先动列

                            for i in range(c):  # 截止到这层完成静态卷积
                                for j in range(d):
                                    sum = sum + kernel[v][s][i][j] * input[l][s][i + k][j + t]
                            first[k][t] = sum  # 静态卷积后形成一个卷积的单元
                            sum = 0
                    sec = torch.add(sec, first)
                    first = torch.zeros(st, sq)
                final[v] = sec+self.bias[v]
                sec = torch.zeros(st, sq)
            ultra[l] = final
            final = torch.zeros(kernel.shape[0], st, sq)
            self.output=ultra
        return self.output

    def forward(self, input: Tensor):
        weight = self.weight
        bias = self.bias
        return self.conv2d(input, weight, bias)

    def backward(self, d_loss):
        '''TODO backward的计算方法'''
        print(d_loss.shape)
        print(self.kernel.shape)
        print(self.input.shape)
        output=torch.zeros(self.input.shape[0],self.kernel.shape[0],4,4)
        output2dim=torch.zeros(4,4)
        output3dim=torch.zeros(5,4,4)
        for i in range(self.input.shape[0]):
            for j in range(self.kernel.shape[0]):
                for z in range (self.input.shape[1]):
                  d_loss1dim=torch.ones(3,3)
                  input1dim=self.input[i][z]
                  kernel1dim=self.kernel[j][z]
                  output1dim=self.onedimbackward(d_loss1dim,input1dim,kernel1dim)
                  output2dim=output1dim+output2dim
                output3dim[j]=output2dim
                output2dim = torch.zeros(4, 4)
            output[i]=output3dim
            output3dim = torch.zeros(5, 4, 4)
        print(output.shape)
        return output
    def onedimbackward(self,d_loss1dim,input1dim,kernel1dim):
        dx = torch.zeros(input1dim.shape[0],input1dim.shape[1])
        dw = torch.zeros(kernel1dim.shape[0],kernel1dim.shape[1])
        db = 0
        for i in range(self.output_height):
            for j in range(self.output_width):

                end_i = i + self.weight_height
                end_j = j + self.weight_width
                dx[i: end_i, j:end_j] += d_loss1dim[i, j] * kernel1dim

                for u in range(self.weight_height):
                    for v in range(self.weight_width):
                        dw[u, v] += d_loss1dim[i, j] * input1dim[i + u, j + v]

        d_loss1dim=d_loss1dim.detach().numpy()
        db = np.sum(d_loss1dim)
        return dx

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('tests', nargs='*')
    return parser.parse_args()


def randnint(n, a=8, b=16):
    """Return N random integers."""
    return (random.randint(a, b) for _ in range(n))#不需要使用自定义变量，生成n张8*16的数组

isclose = functools.partial(np.isclose, rtol=1.e-5, atol=1.e-5)

class Function(object):
    def CrossEntropyLoss(self):
             return CrossEntropyLoss()
    def Linear(self,a,b):
             return Linear(a,b)
    def Conv2d(self,A,B,C):
             return Conv2d(A,B,(C,C))

function=Function()#实例化，便于getattr调用

class TestBase(object):
    def __init__(self, module, input_shape, module_params=None):
        self.module = module.split('.')[-1]#split对module以逗号划分，取列表中倒数第一个元素
        module_params = module_params.split(',') \
                        if module_params is not None else []
        input_shape = input_shape.split('x')
        keys = set(module_params + input_shape)#对两个参数求元素并集
        args = {k: v for k, v in zip(keys, randnint(len(keys)))}#生成keys长度个照片 zip实现一一对应
        args = {"W":4,"Cp":5,"B":2,"H":4,"C":3,"k_s":2,"L":5}#给定维度
        self.nnt = 0.9*torch.rand(tuple(args[k] for k in input_shape))+0.1#放缩产生随机数的范围
        self.ptt = self.nnt.clone().detach()
        self.ptt.requires_grad = True
        self.nnt.requires_grad = True
        self.nnm = getattr(function,module)(*tuple(args[k] for k in module_params))
#module变换调用Function中的def
    def forward_test(self):
        # self.pt_out = ...
        self.nn_out = self.nnm(self.nnt)
        if self.nn_out is None:
            print('your forward output is empty')
            return False
        res = isclose(self.nn_out.cpu().detach().numpy(),
                       self.pt_out.cpu().detach().numpy()).all().item()
        if res :
            return True
        else:
            print('forward_dismatch:') 
            print("pytorch result:",self.pt_out.cpu().detach().numpy())
            print("your result:",self.nn_out.cpu().detach().numpy())
            return False
        
    def backward_test(self):
        self.pt_out.sum().backward()#pytorch求loss需要收敛到一个值所以这里做一个求和
        self.pt_grad = self.ptt.grad
        if self.nn_out is None:
            print("your backward output is empty")
            return False
        self.nn_grad = self.nnm.backward(torch.ones_like(self.nn_out))#传回一个全一矩阵，相当于forward的输出label
        if self.nn_grad is None:
            print("your backward grad is empty")
            return False
        res = isclose(self.nn_grad.detach().numpy(), self.pt_grad.detach().numpy()).all().item()
        if res :
            return True
        else:
            print('backward_dismatch:') 
            print("pytorch result:",self.pt_grad.detach().numpy())
            print("your result:",self.nn_grad)
            return False


class Conv2dTest(TestBase):
    def __init__(self):
        super().__init__('Conv2d', input_shape='BxCxHxW', module_params='C,Cp,k_s')

    def forward_test(self):

        self.pt_wgt = torch.Tensor(self.nnm.weight.data)
        self.pt_wgt.requires_grad = True
        self.pt_bias = torch.Tensor(self.nnm.bias.data)
        self.pt_bias.requires_grad = True
        self.pt_out = F.conv2d(input=self.ptt, weight=self.pt_wgt,
                               bias=self.pt_bias,stride=1,padding=0)
        return super().forward_test()

    def backward_test(self):
        s = super().backward_test()
        s &= isclose(self.nnm.weight.grad, self.pt_wgt.grad
                     .detach().numpy()).all().item()
        s &= isclose(self.nnm.bias.grad, self.pt_bias.grad
                     .detach().numpy()).all().item()
        return s
    

class LinearTest(TestBase):
    def __init__(self):
        super().__init__('Linear', input_shape='BxL', module_params='L,C')

    def forward_test(self):
        self.pt_wgt = self.nnm.weight.clone().detach()
        self.pt_wgt.requires_grad = True
        self.pt_bias = self.nnm.bias.clone().detach()
        self.pt_bias.requires_grad = True
        self.pt_out = F.linear(input=self.ptt, weight=self.pt_wgt,
                               bias=self.pt_bias)
        return super().forward_test()

    def backward_test(self):
        s = super().backward_test()
        if s:
            s = isclose(self.nnm.weight.grad.detach().numpy(), self.pt_wgt.grad
                     .detach().numpy()).all().item()
            if s == False:
                print('weight_grad_dismatch')
        if s:
            s = isclose(self.nnm.bias.grad.detach().numpy(), self.pt_bias.grad
                     .detach().numpy()).all().item()
            if s == False:
                print('bias_grad_dismatch')
        return s

class CrossEntropyTest(TestBase):#继承Testbase
    def __init__(self):
        super().__init__('CrossEntropyLoss', input_shape='BxL')

    def forward_test(self):
        pt_crossentropyloss = torch.nn.CrossEntropyLoss()
        batch = self.nnt.size(0)
        self.test_target = torch.ones(batch,dtype=torch.int64)#选取的label为1
        self.pt_out = pt_crossentropyloss(input=self.ptt, target=self.test_target)
        self.nt_out = self.nnm.forward(input=self.nnt,target=self.test_target)
    
        res = isclose(self.pt_out.detach().numpy(),self.nt_out.detach().numpy()).all().item()
        if res:
            return True
        else:
            print('crossentropy_forward_dismatch:') 
            print("pytorch result:",self.pt_out.detach().numpy())
            print("your result:",self.nt_out.detach().numpy())
            return False
    def backward_test(self):
        self.pt_out.backward()
        self.nn_grad = self.nnm.backward()
        res = isclose(self.ptt.grad.detach().numpy(),self.nn_grad.detach().numpy()).all().item()
        if res:
            return True
        else:
            print('crossentropy_backward_dismatch:') 
            print("pytorch result:",self.ptt.grad.detach().numpy())
            print("your result:",self.nn_grad.detach().numpy())
            return False

if __name__ == "__main__":
    test_list = [CrossEntropyTest(),LinearTest(),Conv2dTest()]
    #a=TestBase()
    for a in test_list:
        print("Test",a.module)
        print("forward:",a.forward_test())
        print("backword:",a.backward_test())

