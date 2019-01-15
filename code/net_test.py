import net
import torch
from torch.autograd import Variable
from vis_net import make_dot

from graphviz import Digraph
import os
os.environ["PATH"] += os.pathsep + 'D:/Program Files (x86)/Graphviz2.38/bin/'

if __name__ == '__main__':
    z = Variable(torch.rand(1,3,127,127))
    x = Variable(torch.rand(1,3,255,255))
    # net = net.RDN()
    # print(net.eval())


    snet = net.DenseSiameseRPN()
    print(snet.eval())
    zf = snet.features(z)
    xf = snet.features(x)
    print(zf.size())
    print(xf.size())
    out = snet(z,x)
    # # out = net(x)
    print(out[0].size())
    g1 = make_dot(out[0])
    g1.view()
