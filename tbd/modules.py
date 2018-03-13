# DISTRIBUTION STATEMENT A. Approved for public release: distribution unlimited.
#
# This material is based upon work supported by the Assistant Secretary of Defense for Research and
# Engineering under Air Force Contract No. FA8721-05-C-0002 and/or FA8702-15-D-0001. Any opinions,
# findings, conclusions or recommendations expressed in this material are those of the author(s) and
# do not necessarily reflect the views of the Assistant Secretary of Defense for Research and
# Engineering.
#
# © 2017 Massachusetts Institute of Technology.
#
# MIT Proprietary, Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)
#
# The software/firmware is provided to you on an As-Is basis
#
# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or
# 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are
# defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than
# as specifically authorized by the U.S. Government may violate any copyrights that exist in this
# work.

import torch
import torch.nn as nn
import torch.nn.functional as F


class AndModule(nn.Module):
    """ A neural module that (basically) performs a logical and.

    Extended Summary
    ---------------- 
    An :class:`AndModule` is a neural module that takes two input attention masks and (basically)
    performs a set intersection. This would be used in a question like "What color is the cube to
    the left of the sphere and right of the yellow cylinder?" After localizing the regions left of
    the sphere and right of the yellow cylinder, an :class:`AndModule` would be used to find the
    intersection of the two. Its output would then go into an :class:`AttentionModule` that finds
    cubes.
    """
    def forward(self, attn1, attn2):
        out = torch.min(attn1, attn2)
        return out


class OrModule(nn.Module):
    """ A neural module that (basically) performs a logical or.

    Extended Summary
    ----------------
    An :class:`OrModule` is a neural module that takes two input attention masks and (basically)
    performs a set union. This would be used in a question like "How many cubes are left of the
    brown sphere or right of the cylinder?" After localizing the regions left of the brown sphere
    and right of the cylinder, an :class:`OrModule` would be used to find the union of the two. Its
    output would then go into an :class:`AttentionModule` that finds cubes.
    """
    def forward(self, attn1, attn2):
        out = torch.max(attn1, attn2)
        return out


class AttentionModule(nn.Module):
    """ A neural module that takes a feature map and attention, attends to the features, and 
    produces an attention.

    Extended Summary
    ----------------
    An :class:`AttentionModule` takes input features and an attention and produces an attention. It
    multiplicatively combines its input feature map and attention to attend to the relevant region
    of the feature map. It then processes the attended features via a series of convolutions and
    produces an attention mask highlighting the objects that possess the attribute the module is
    looking for.

    For example, an :class:`AttentionModule` may be tasked with finding cubes. Given an input
    attention of all ones, it will highlight all the cubes in the provided input features. Given an
    attention mask highlighting all the red objects, it will produce an attention mask highlighting
    all the red cubes.

    Attributes
    ----------
    dim : int
        The number of channels of each convolutional filter.
    """
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1) 
        self.conv3 = nn.Conv2d(dim, 1, kernel_size=1, padding=0)
        torch.nn.init.kaiming_normal(self.conv1.weight)
        torch.nn.init.kaiming_normal(self.conv2.weight)
        torch.nn.init.kaiming_normal(self.conv3.weight)
        self.dim = dim

    def forward(self, feats, attn):
        attended_feats = torch.mul(feats, attn.repeat(1, self.dim, 1, 1))
        out = F.relu(self.conv1(attended_feats))
        out = F.relu(self.conv2(out))
        out = F.sigmoid(self.conv3(out))
        return out


class QueryModule(nn.Module):
    """ A neural module that takes as input a feature map and an attention and produces a feature
    map as output.

    Extended Summary
    ----------------
    A :class:`QueryModule` takes a feature map and an attention mask as input. It attends to the
    feature map via an elementwise multiplication with the attention mask, then processes this
    attended feature map via a series of convolutions to extract relevant information.

    For example, a :class:`QueryModule` tasked with determining the color of objects would output a
    feature map encoding what color the attended object is. A module intended to count would output
    a feature map encoding the number of attended objects in the scene.

    Attributes
    ----------
    dim : int
        The number of channels of each convolutional filter.
    """
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        torch.nn.init.kaiming_normal(self.conv1.weight)
        torch.nn.init.kaiming_normal(self.conv2.weight)
        self.dim = dim

    def forward(self, feats, attn):
        attended_feats = torch.mul(feats, attn.repeat(1, self.dim, 1, 1))
        out = F.relu(self.conv1(attended_feats))
        out = F.relu(self.conv2(out))
        return out


class RelateModule(nn.Module):
    """ A neural module that takes as input a feature map and an attention and produces an attention
    as output.

    Extended Summary
    ----------------
    A :class:`RelateModule` takes input features and an attention and produces an attention. It
    multiplicatively combines the attention and the features to attend to a relevant region, then
    uses a series of dilated convolutional filters to indicate a spatial relationship to the input
    attended region.

    Attributes
    ----------
    dim : int
        The number of channels of each convolutional filter.
    """
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, dilation=1)  # receptive field 3
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=2, dilation=2)  # 7
        self.conv3 = nn.Conv2d(dim, dim, kernel_size=3, padding=4, dilation=4)  # 15
        self.conv4 = nn.Conv2d(dim, dim, kernel_size=3, padding=8, dilation=8)  # 31 -- full image
        self.conv5 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, dilation=1)
        self.conv6 = nn.Conv2d(dim, 1, kernel_size=1, padding=0)
        torch.nn.init.kaiming_normal(self.conv1.weight)
        torch.nn.init.kaiming_normal(self.conv2.weight)
        torch.nn.init.kaiming_normal(self.conv3.weight)
        torch.nn.init.kaiming_normal(self.conv4.weight)
        torch.nn.init.kaiming_normal(self.conv5.weight)
        torch.nn.init.kaiming_normal(self.conv6.weight)
        self.dim = dim

    def forward(self, feats, attn):
        feats = torch.mul(feats, attn.repeat(1, self.dim, 1, 1))
        out = F.relu(self.conv1(feats))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.relu(self.conv5(out))
        out = F.sigmoid(self.conv6(out))
        return out


class SameModule(nn.Module):
    """ A neural module that takes as input a feature map and an attention and produces an attention
    as output.

    Extended Summary
    ----------------
    A :class:`SameModule` takes input features and an attention and produces an attention. It
    determines the index of the maximally-attended object, extracts the feature vector at that
    spatial location, then performs a cross-correlation at each spatial location to determine which
    other regions have this same property. This correlated feature map then goes through a
    convolutional block whose output is an attention mask.

    As an example, this module can be used with the CLEVR dataset to perform the `same_shape`
    operation, which will highlight every region of an image that shares the same shape as an object
    of interest (excluding the original object).

    Attributes
    ----------
    dim : int
        The number of channels in the input feature map.
    """
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim+1, 1, kernel_size=1)
        torch.nn.init.kaiming_normal(self.conv.weight)
        self.dim = dim

    def forward(self, feats, attn):
        size = attn.size()[2]
        the_max, the_idx = F.max_pool2d(attn, size, return_indices=True)
        attended_feats = feats.index_select(2, the_idx[0, 0, 0, 0] / size)
        attended_feats = attended_feats.index_select(3, the_idx[0, 0, 0, 0] % size)
        x = torch.mul(feats, attended_feats.repeat(1, 1, size, size))
        x = torch.cat([x, attn], dim=1)
        out = F.sigmoid(self.conv(x))
        return out


class ComparisonModule(nn.Module):
    """ A neural module that takes as input two feature maps and produces a feature map as output.

    Extended Summary
    ----------------
    A :class:`ComparisonModule` takes two feature maps as input and concatenates these. It then
    processes the concatenated features and produces a feature map encoding whether the two input
    feature maps encode the same property.

    This block is useful in making integer comparisons, for example to answer the question, ``Are
    there more red things than small spheres?'' It can also be used to determine whether some
    relationship holds of two objects (e.g. they are the same shape, size, color, or material).

    Attributes
    ----------
    dim : int
        The number of channels of each convolutional filter.
    """
    def __init__(self, dim):
        super().__init__()
        self.projection = nn.Conv2d(2*dim, dim, kernel_size=1, padding=0)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        torch.nn.init.kaiming_normal(self.conv1.weight)
        torch.nn.init.kaiming_normal(self.conv2.weight)

    def forward(self, in1, in2):
        out = torch.cat([in1, in2], 1)
        out = F.relu(self.projection(out))
        out = F.relu(self.conv1(out))
        out = F.relu(self.conv2(out))
        return out
