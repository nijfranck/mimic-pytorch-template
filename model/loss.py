import torch.nn.functional as F
import torch


def nll_loss(output, target):
    return F.nll_loss(output, target)


def cross_entropy(output, target):
    return F.cross_entropy(output, target)


# def cross_entropy_weights(output, target, weights=torch.FloatTensor([1, 4]).cuda()):
#     return F.cross_entropy(output, target, weight=weights).to(torch.device('cuda:0'))
