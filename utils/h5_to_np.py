# Provides functions for converting hdf5 files to npy files
#
# This script can be invoked with commandline arguments. E.g.
# `python h5-to-np.py -q 'path/to/questions.h5' -f 'path/to/features.h5' `
#
# License: MIT. See LICENSE.txt for the full license.
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

import h5py
import numpy as np
import argparse
from pathlib import Path

__all__ = ['questions_to_npy', 'images_to_npy', 'features_to_npy']

def _verbose_save(h5_file, h5_key, dest_dir, dest_name, dtype):
    """ Save array stored in hdf5 dataset to a .npy file: {dest_dir}/{dest_name}.npy

        Parameters
        ----------
        h5_file : h5py.File
            Read-mode hdf5 file object.

        h5_key : str
            Item-retrieval key for hdf5 dataset.

        dest_dir : Union[pathlib.Path, str]
            Path to the directory in which the .npy file will be saved.

        dest_name : str
            The file will be `dest_name`.npy

        dtype : numpy.dtype
            Data type of the array being saved.

        Returns
        -------
        None
    """
    if dest_name.endswith(".npy"):
        dest_name = dest_name[:-4]
    dest = Path(dest_dir)
    print('Saving {} as npy...'.format(dest_name))
    np.save(dest / dest_name, np.asarray(h5_file[h5_key], dtype=dtype))
    print('Saved {} as {}'.format(dest_name, (dest / (dest_name + '.npy')).absolute()))

def questions_to_npy(hdf5_path, dest_dir='.', dtype=np.int64):
    """ Reads from a hdf5-dataset encoded questions, image indices, and optionally,
        the programs and image indices, and saves the arrays to .npy files.

        Parameters
        ----------
        hdf5_path : Union[pathlib.Path, str]
            Path to the hdf5 file to be read.

        dest_dir : Union[pathlib.Path, str]
            Path to the destination directory. Default: present working directory.

        dtype : numpy.dtype
            Data type of the array being saved. Default: np.int64.

        Returns
        -------
        None

        Notes
        -----
        The output files are saved to:
         - {dest_dir}/questions.npy
         - {dest_dir}/image_indices.npy
         - {dest_dir}/programs.npy
         - {dest_dir}/answers.npy
    """
    with h5py.File(hdf5_path, mode='r') as F:
        _verbose_save(F, 'questions', dest_dir, 'questions', dtype)
        _verbose_save(F, 'image_idxs', dest_dir, 'image_indices', dtype)
        
        if 'programs' in F:
            _verbose_save(F, 'programs', dest_dir, 'programs', dtype)
    
        if 'answers' in F:
            _verbose_save(F, 'answers', dest_dir, 'answers', dtype)

def images_to_npy(hdf5_path, dest_dir='.', dtype=np.float32):
    """ Reads image data from a hdf5-dataset saves them to a .npy file.

        Parameters
        ----------
        hdf5_path : Union[pathlib.Path, str]
            Path to the hdf5 file to be read.

        dest_dir : Union[pathlib.Path, str]
            Path to the destination directory. Default: present working directory.

        dtype : numpy.dtype
            Data type of the array being saved. Default: np.float32.

        Returns
        -------
        None

        Notes
        -----
        The output file is saved to:
         - {dest_dir}/images.npy
    """
    with h5py.File(hdf5_path, mode='r') as F:
        _verbose_save(F, 'images', dest_dir, 'images', dtype)

def features_to_npy(hdf5_path, dest_dir='.', dtype=np.float32):
    """ Reads feature data from a hdf5-dataset saves them to a .npy file.

        Parameters
        ----------
        hdf5_path : Union[pathlib.Path, str]
            Path to the hdf5 file to be read.

        dest_dir : Union[pathlib.Path, str]
            Path to the destination directory. Default: present working directory.

        dtype : numpy.dtype
            Data type of the array being saved. Default: np.float32.

        Returns
        -------
        None

        Notes
        -----
        The output file is saved to:
         - {dest_dir}/features.npy"""
    with h5py.File(hdf5_path, mode='r') as F:
        _verbose_save(F, 'features', dest_dir, 'features', dtype)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--questions', '-q', required=True,
                        help='HDF5 file containing the questions')

    parser.add_argument('--features', '-f', required=True,
                        help='HDF5 file containing the extracted image features')

    parser.add_argument('--images', '-i', required=False, default=None,
                        help='HDF5 file containing the images')

    parser.add_argument('--destination', '-d', required=False, default='.',
                        help='The directory to write the numpy files to')

    args = parser.parse_args()

    questions_to_npy(args.questions, args.destination)
    features_to_npy(args.features, args.destination)

    if args.images is not None:
        images_to_npy(args.images, args.destination)

    print('Finished!')
