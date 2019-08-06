import os
import sys
import unittest
import argparse
os.chdir('..')
from template_lib import utils


class TestingTrainSearch(unittest.TestCase):

  def test_cnn_cifar10_train_search(self):
    """
    Usage:
        source activate Peterou_torch_py36
        export CUDA_VISIBLE_DEVICES=4
        export PORT=6010
        export TIME_STR=1
        export PYTHONPATH=../../submodule:../..
        python -c "import test_cnn; \
        test_cnn.TestingTrainSearch().test_cnn_cifar10_train_search()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6011'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

    # func name
    outdir = os.path.join('results', sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
            --config configs/cnn_cifar10.yaml
            --command cnn_cifar10_train_search
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
    import search
    search.run(args, myargs)
    input('End %s' % outdir)
    return