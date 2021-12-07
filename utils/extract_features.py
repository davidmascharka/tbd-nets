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
import numpy as np
from PIL import Image
from scipy.misc import imread
from torchvision.models import resnet101

def load_feature_extractor(model_stage=2):
    """ Load the appropriate parts of ResNet-101 for feature extraction.

    Parameters
    ----------
    model_stage : Integral
        The stage of ResNet-101 from which to extract features.
        For 28x28 feature maps, this should be 2. For 14x14 feature maps, 3.

    Returns
    -------
    torch.nn.Sequential
        The feature extractor (ResNet-101 at `model_stage`)

    Notes
    -----
    This function will download ResNet-101 if it is not already present through torchvision.
    """
    model = resnet101(pretrained=True)
    layers = [model.conv1, model.bn1, model.relu, model.maxpool]
    layers += [getattr(model, 'layer{}'.format(i+1)) for i in range(model_stage)]
    model = torch.nn.Sequential(*layers)
    if torch.cuda.is_available():
        model.cuda()

    return model.eval()


def extract_image_feats(img_path, model):
    """ Extract image features from the image at `img_path` using `model`.

    Parameters
    ----------
    img_path : Union[pathlib.Path, str]
        The path to the image file.

    model : torch.nn.Module
        The feature extractor to use.

    Returns
    -------
    Tuple[numpy.ndarray, torch.Tensor]
        The image and image features extracted from `model`
    """
    # read in the image and transform it to shape (1, 3, 224, 224)
    path = str(img_path) # to handle pathlib
    img = imread(path, mode='RGB')
    img = np.array(Image.fromarray(img).resize((224, 224), resample=Image.BICUBIC))
    img = img.transpose(2, 0, 1)[None]

    # use ImageNet statistics to transform the data
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.224]).reshape(1, 3, 1, 1)
    img_tensor = torch.FloatTensor((img / 255 - mean) / std)

    # push to the GPU if possible
    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()

    return (img.squeeze().transpose(1, 2, 0), model(img_tensor))
