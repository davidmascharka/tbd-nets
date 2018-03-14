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

from pathlib import Path
from urllib.request import urlretrieve
import sys
import argparse


def _download_info(blocks_so_far, block_size, total_size):
    percent = int(100*blocks_so_far*block_size / total_size)
    print('Downloaded {}%'.format(percent), end='\r')
    if percent == 100:
        print('Finished downloading')

def download(fnames):
    """ Utility to download TbD-Net models.

    Parameters
    ----------
    fnames : Union[str, Sequence]
        The file name(s) of the models to download.

    Notes
    -----
    Available model options (fnames) are as follows:
      - 'clevr.pt' : trained on CLEVR with 14x14 feature maps
      - 'cogent-no-finetune.pt' : the same as 'clevr.pt' trained on CoGenT A without fine-tuning
      - 'cogent-finetuned.pt' : the same as above trained on CoGenT A and fine-tuned on CoGenT B
      - 'clevr-reg.pt' : trained on CLEVR with 14x14 feature maps using regularization
      - 'cogent-no-finetune-reg.pt' : the same as above trained on CoGenT A without fine-tuning
      - 'cogent-finetuned-reg.pt' : the same as above trained on CoGenT A and fine-tuned on B
      - 'clevr-reg-hres.pt' : trained on CLEVR with 28x28 feature maps and regularization
      - 'program_generator.pt' : the program generator; see 'generate_programs.py'
    """
    download_path = Path('./models')
    if not download_path.exists() or not download_path.is_dir():
        print('The directory \'models\' does not exist!')
        print('Please ensure you are in the top level of the visual-attention-networks repository')
        print('  and that the \'models\' directory exists')
        sys.exit()

    server_url = 'https://github.com/davidmascharka/tbd-nets/releases/download/v1.0/'
    if isinstance(fnames, str): # a single file
        fnames = [fnames]
    for fname in fnames:
        if (download_path / fname).exists():
            print('Skipping {}: the file already exists'.format(fname))
            continue

        print('Downloading {}'.format(fname))
        urlretrieve(server_url + fname, str((download_path/fname).absolute()), _download_info)
    print('Finished')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_str = 'Which models to download. Options: original, reg, hres, all (default: hres)' + \
               ' Original: no regularization. Reg: regularized with lambda = 2.5e-07.' + \
               ' Hres: high-resolution feature maps with regularization. All: all'
    parser.add_argument('--models', '-m', required=False, default='hres',
                        choices=['original', 'reg', 'hres', 'all'], help=help_str)
    
    args = parser.parse_args()

    original_fnames = ['clevr.pt', 'cogent-no-finetune.pt', 'cogent-finetuned.pt']
    reg_fnames = ['clevr-reg.pt', 'cogent-no-finetune-reg.pt', 'cogent-finetuned-reg.pt']
    hres_fnames = ['clevr-reg-hres.pt']
    
    if args.models in {'hres', 'all'}:
        download(hres_fnames)
    if args.models in {'reg', 'all'}:
        download(reg_fnames)
    if args.models in {'original', 'all'}:
        download(original_fnames)
