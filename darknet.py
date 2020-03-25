#coding=utf-8
from __future__ import division #导入未来精确除法函数
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer,self).__init__() #首先找到EmptyLayer的父类（比如是类nn.Module），然后把类EmptyLayer的对象self转换为类nn.Module的对象，然后“被转换”的类nn.Module对象调用类nn.Module对象自己的__init__函数.

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer,self).__init__()
        self.anchors = anchors


def parse_cfg(cfgfile):
    """
    Takes a configuration file

    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list

    """
    file = open(cfgfile, 'r')                             #open(name[, mode[, buffering]]) mode:'w', 'r' type(file)=str?
    lines = file.read().split('\n')                      #str.split(str="", num=string.count(str))[num] 返回以str为分隔的list，并去第num个切片。 read从文件指针位置读到文件结尾。将file的内容按行分隔开
    lines = [x for x in lines if len(x) > 0]        #get rid of empty lines
    lines = [x for x in lines if x[0]!= '#']         #get rid of comments
    lines = [x.lstrip().rstrip() for x in lines]     #get rid of fringe white
    # file is a str, lines is a list

    # seprate all the layer
    block = {}
    blocks = []

    for line in lines:
        if line[0] == '[':                           #this marks a new layer
            if len(block) != 0:                    # if the block is not empty, it implies the block has store the previous block in cfg
                blocks.append(block)          #append this block(dict) into the list blocks
                block = {}     #init the block(dict)
            block["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split("=") #解包
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)   #store the last block(dict)
    return blocks






def create_modules(blocks):
    net_info = blocks[0]  # Captures the information about the input and pre-processing
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    for index, x in enumerate(blocks[1:]):  # enumerate#enumerate 返回列表中的每一个值，并在前面加上一个标号
        module = nn.Sequential()  # for every block in cfg, it has various layer. we use a sequential to store this multiple layer
        if (x["type"] == "convolutional"):
            # get the info from the layer
            activation = x["activation"]
            try:   #先运行try后代码，若正确，则执行else；若错误，则执行except
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True
            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])
            if padding != 0:
                pad = (kernel_size - 1) // 2  # integrated multiplication, return the biggist int samller than the result
            else:
                pad = 0

            # add a convolutional layer
            conv = nn.Conv2d(prev_filters, filters, kernel_size=kernel_size, stride=stride, padding=pad, bias=bias)  # class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
            module.add_module("conv_{0}".format(index), conv)  # add_module(name, module) name gives this layer a new name. {0}.format(index), the "0" will shows the "index"

            # Add the Batch Norm Layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)  # class torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True)
                # num_features：来自期望输出的特征数，该期望输入的大小为'batch_size x num_features x height x width'eps： 为保证数值稳定性（分母不能趋近或取0）,给分母加上的值。默认为1e-5。
                module.add_module("batch_norm_{0}".format(index), bn)

            # Check the activation.
            # It is either Linear or a Leaky ReLU for YOLO
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace=True)  # torch.nn.LeakyReLU(negative_slope=0.01, inplace=False)
                module.add_module("leaky_{0}".format(index), activn)

        # If it's an upsampling layer
        # We use Bilinear2dUpsampling
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            # torch.nn.Upsample(size=None, scale_factor=None, mode='nearest', align_corners=None)
            # size(tuple, optional) – 一个包含两个整数的元组(H_out, W_out)指定了输出的长宽
            # scale_factor(int, optional) – 长和宽的一个乘子
            module.add_module('upsamle_{0}'.format(index), upsample)

        elif (x["type"] == "route"):
            x['layers'] = x['layers'].split(',')  # x{'layer':['-4','64']}
            # Start  of a route
            start = int(x['layers'][0])
            # end, if there exists one.
            try:
                end = int(x['layers'][1])
            except:
                end = 0
            # Positive anotation
            if start > 0:
                start = start - index  #将start和end变为在当前层（index）之前多少层（负数）的形式
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module('route_{0}'.format(index), route)
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]

        elif (x["type"] == "shortcut"):
            shortcut = EmptyLayer()
            module.add_module("shortcut_{0}".format(index), shortcut)

        elif (x["type"] == "yolo"):
            masks = x["mask"].split(",")
            masks = [int(i) for i in masks]

            anchors = x["anchors"].split(",")
            anchors = [int(i) for i in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in masks]

            detection = DetectionLayer(anchors)
            module.add_module('DetectionLayer_{0}'.format(index), detection)

        module_list.append(module)
        output_filters.append(filters)
        prev_filters = filters

    return (net_info, module_list)


blocks = parse_cfg("cfg/yolov3.cfg")
print(create_modules(blocks))
