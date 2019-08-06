import foolbox
import torch
import torchvision.models as models
import numpy as np
import os
import sys
import unittest
import argparse

os.chdir('..')
from template_lib import utils


class Testing_foolbox(unittest.TestCase):

  def test_untargeted_adversarial(self):
    """
    Usage:
        source activate Peterou_torch_0_3_1_tv_0_2_0
        export CUDA_VISIBLE_DEVICES=2,3,4,5
        export PORT=6006
        export TIME_STR=1
        export PYTHONPATH=../../submodule:..
        python -c "import test_foolbox; \
        test_foolbox.Testing_foolbox().test_untargeted_adversarial()"
    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6011'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

    # func name
    outdir = os.path.join('results/foolbox', sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
            --config configs/foolbox.yaml 
            --command untargeted_adversarial
            --resume False --resume_path None
            --resume_root None
            """
      parser = utils.args_parser.build_parser()
      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      args.CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']
      args = utils.config_utils.DotDict(vars(args))
      return args, argv_str
    args, argv_str = build_args()

    args.outdir = outdir
    args, myargs = utils.config.setup_args_and_myargs(args=args, myargs=myargs)

    # instantiate the model
    resnet18 = models.resnet18(pretrained=True).eval()
    if torch.cuda.is_available():
      resnet18 = resnet18.cuda()
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    fmodel = foolbox.models.PyTorchModel(
      resnet18, bounds=(0, 1), num_classes=1000, preprocessing=(mean, std))

    # get source image and label
    image, label = foolbox.utils.imagenet_example(data_format='channels_first')
    image = image / 255.  # because our model expects values in [0, 1]

    print('label', label)
    print('predicted class', np.argmax(fmodel.forward_one(image)))

    # apply attack on source image
    attack = foolbox.attacks.FGSM(fmodel)
    adversarial = attack(image, label)

    print('adversarial class', np.argmax(fmodel.forward_one(adversarial)))

    return









