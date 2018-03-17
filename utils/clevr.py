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

import numpy as np
import h5py
import warnings
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

__all__ = ['invert_dict', 'load_vocab', 'clevr_collate', 'ClevrDataLoaderH5',
           'ClevrDataLoaderNumpy', 'ClevrDataset']

def invert_dict(d):
    """ Utility for swapping keys and values in a dictionary.
    
    Parameters
    ----------
    d : Dict[Any, Any]
    
    Returns
    -------
    Dict[Any, Any]
    """
    return {v: k for k, v in d.items()}


def load_vocab(path):
    """ Load the vocabulary file.

    Parameters
    ----------
    path : Union[str, pathlib.Path]
        Path to the vocabulary json file.

    Returns
    -------
    vocab : Dict[str, Dict[Any, Any]]
        The vocabulary. See the extended summary for details on the values.

    Extended Summary
    ----------------
    The vocabulary object contains the question, program, and answer tokens and their respective
    image indices. It is a dictionary of dictionaries. Its contents are:

    - question_idx_to_token : Dict[int, str]
        A mapping from question index to question word.

    - program_idx_to_token : Dict[int, str]
        A mapping from program index to module name/logical operation (e.g. filter_color[red]).

    - answer_idx_to_token : Dict[int, str]
        A mapping from answer index to answer word.

    - question_token_to_idx : Dict[str, int]
        A mapping from question word to index.

    - program_token_to_idx : Dict[str, int]
        A mapping from program index to module description.

    - answer_token_to_idx : Dict[str, int]
        A mapping from answer word to index.
    """
    path = str(path)  # in case we get a pathlib.Path
        
    with open(path, 'r') as f:
        vocab = json.load(f)
        vocab['question_idx_to_token'] = invert_dict(vocab['question_token_to_idx'])
        vocab['program_idx_to_token'] = invert_dict(vocab['program_token_to_idx'])
        vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])

    # our answers format differs from that of Johnson et al.
    answers = ['blue', 'brown', 'cyan', 'gray', 'green', 'purple', 'red', 'yellow',
               'cube', 'cylinder', 'sphere',
               'large', 'small',
               'metal', 'rubber',
               'no', 'yes',
               '0', '1', '10', '2', '3', '4', '5', '6', '7', '8', '9']
    vocab['answer_idx_to_token'] = dict(zip(range(len(answers)), answers))
    vocab['answer_token_to_idx'] = dict(zip(answers, range(len(answers))))
    return vocab


def clevr_collate(batch):
    """ Collate a batch of data."""
    transposed = list(zip(*batch))
    question_batch = default_collate(transposed[0])
    image_batch = transposed[1]
    if any(img is not None for img in image_batch):
        image_batch = default_collate(image_batch)
    feat_batch = transposed[2]
    if any(f is not None for f in feat_batch):
        feat_batch = default_collate(feat_batch)
    answer_batch = default_collate(transposed[3]) if transposed[3][0] is not None else None
    program_seq_batch = transposed[4]
    if transposed[4][0] is not None:
        program_seq_batch = default_collate(transposed[4])
    return [question_batch, image_batch, feat_batch, answer_batch, program_seq_batch]


class ClevrDataset(Dataset):
    """ Holds a handle to the CLEVR dataset.

    Extended Summary
    ----------------
    A :class:`ClevrDataset` holds a handle to the CLEVR dataset. It loads a specified subset of the
    questions, their image indices and extracted image features, the answer (if available), and
    optionally the images themselves. This is best used in conjunction with a
    :class:`ClevrDataLoaderNumpy` of a :class:`ClevrDataLoaderH5`, which handle loading the data.
    """
    def __init__(self, questions, image_indices, programs, features, answers, images=None,
                 indices=None):
        """ Initialize a ClevrDataset object.
        
        Parameters
        ----------
        questions : Union[numpy.ndarray, h5py.File]
            Object holding the questions.

        image_indices : Union[numpy.ndarray, h5py.File]
            Object holding the image indices.

        programs : Union[numpy.ndarray, h5py.File]
            Object holding the programs, or None.

        features : Union[numpy.ndarray, h5py.File]
            Object holding the extracted image features.

        answers : Union[numpy.ndarray, h5py.File]
            Object holding the answers, or None.

        images : Union[numpy.ndarray, h5py.File], optional
            Object holding the images, or None.

        indices : Sequence[int], optional
            The indices of the questions to load, or None.
        """
        assert len(questions) == len(image_indices) == len(programs) == len(answers), \
            'The questions, image indices, programs, and answers are not all the same size!'

        # questions, image indices, programs, and answers are small enough to load into memory
        self.all_questions = torch.LongTensor(np.asarray(questions))
        self.all_image_idxs = torch.LongTensor(np.asarray(image_indices))
        self.all_programs = torch.LongTensor(np.asarray(programs))
        self.all_answers = torch.LongTensor(np.asarray(answers)) if answers is not None else None

        # features and images are not small enough to always load
        self.features = features
        self.images = images

        if indices is not None:
            indices = torch.LongTensor(np.asarray(indices))
            self.all_questions = self.all_questions[indices]
            self.all_image_idxs = self.all_image_idxs[indices]
            self.all_programs = self.all_programs[indices]
            self.all_answers = self.all_answers[indices]

    def __getitem__(self, index):
        question = self.all_questions[index]
        image_idx = self.all_image_idxs[index]
        answer = self.all_answers[index] if self.all_answers is not None else None
        program_seq = self.all_programs[index]
        image = None
        if self.images is not None:
            image = torch.FloatTensor(np.asarray(self.image_np[image_idx]))
        feats = torch.FloatTensor(np.asarray(self.features[image_idx]))

        return (question, image, feats, answer, program_seq)

    def __len__(self):
        return len(self.all_questions)


class ClevrDataLoaderNumpy(DataLoader):
    """ Loads the CLEVR dataset from npy files.

    Extended Summary
    ----------------
    Loads the data for, and handles construction of, a :class:`ClevrDataset`. This object can then
    be used to iterate through batches of data for training, validation, or testing.
    """
    def __init__(self, **kwargs):
        """ Initialize a ClevrDataLoaderNumpy object.

        Parameters
        ----------
        question_np : Union[pathlib.Path, str]
            Path to the numpy file holding the questions.

        feature_np : Union[pathlib.Path, str]
            Path to the numpy file holding the extracted image features.

        image_idx_np : Union[pathlib.Path, str]
            Path to the numpy file holding the indices of each question's corresponding image.

        program_np : Union[pathlib.Path, str]
            Path to the numpy file holding the programs the module network should compose, or None.

        answer_np : Union[pathlib.Path, str]
            Path to the numpy file holding the answers to each question, or None.

        image_np : Union[pathlib.Path, str], optional
            Path to the numpy file holding the raw images.

        shuffle : bool, optional
            Whether to shuffle the data.

        indices : Sequence[int], optional
            The question indices to load, or None.
        """
        # The questions, image features, image indices, programs, and answers are required.
        if 'question_np' not in kwargs:
            raise ValueError('Must give question_np')
        if 'feature_np' not in kwargs:
            raise ValueError('Must give feature_np')
        if 'image_idx_np' not in kwargs:
            raise ValueError('Must give image_idx_np')
        if 'program_np' not in kwargs:
            raise ValueError('Must give program_np')
        if 'answer_np' not in kwargs:
            raise ValueError('Must give answer_np')

        # We're mmapping the image features because they aren't small enough for everybody to load.
        # If you have about 65 GB of memory available, feel free to remove the mmap_mode argument to
        # load the entire dataset into memory. This will eliminate some overhead.
        feature_np_path = str(kwargs.pop('feature_np'))
        print('Reading features from ', feature_np_path)
        feature_np = np.load(feature_np_path, mmap_mode='r')

        # The same goes for the images. If you want to load all the images for some reason, that can
        # be done by removing the mmap_mode argument below.
        image_np = None
        if 'image_np' in kwargs:
            image_np_path = str(kwargs.pop('image_np'))
            print('Reading images from ', image_np_path)
            image_np = np.load(image_np_path, mmap_mode='r')

        # The question, image, program, and answer arrays are small enough to load into memory, so
        # we directly load them here.
        question_np_path = str(kwargs.pop('question_np'))
        print('Reading questions from ', question_np_path)
        question_np = np.load(question_np_path)

        image_idx_np_path = str(kwargs.pop('image_idx_np'))
        print('Reading image indices from', image_idx_np_path)
        image_idx_np = np.load(image_idx_np_path)

        program_np_path = str(kwargs.pop('program_np'))
        print('Reading programs from', program_np_path)
        program_np = np.load(program_np_path)

        answer_np_path = str(kwargs.pop('answer_np'))
        print('Reading answers from', answer_np_path)
        answer_np = np.load(answer_np_path) if answer_np_path is not None else None

        indices = None
        if 'indices' in kwargs:
            indices = kwargs.pop('indices')

        if 'shuffle' not in kwargs:
            # Be nice, and make sure the user knows they aren't shuffling the data
            warnings.warn('\n\n\tYou have not provided a \'shuffle\' argument to the data '
                      'loader.\n\tBe aware that the default behavior is to NOT shuffle the data.\n')

        self.dataset = ClevrDataset(question_np, image_idx_np, program_np, feature_np, answer_np,
                                    image_np, indices)
        kwargs['collate_fn'] = clevr_collate
        super().__init__(self.dataset, **kwargs)

    def close(self):
        # numpy handles everything here
        return

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class ClevrDataLoaderH5(DataLoader):
    """ Loads the CLEVR dataset from HDF5 files.

    Extended Summary
    ----------------
    Loads the data for, and handles construction of, a :class:`ClevrDataset`. This object can then
    be used to iterate through batches of data for training, validation, or testing.
    """
    def __init__(self, **kwargs):
        """ Initialize a ClevrDataLoaderH5 object.

        Parameters
        ----------
        question_h5 : Union[pathlib.Path, str]
            Path to the HDF5 file holding the questions, image indices, programs, and answers.

        feature_h5 : Union[pathlib.Path, str]
            Path to the HDF5 file holding the extracted image features.

        image_h5 : Union[pathlib.Path, str], optional
            Path to the HDF5 file holding the raw images.

        shuffle : bool, optional
            Whether to shuffle the data.

        indices : Sequence[int], optional
            The question indices to load, or None.
        """
        if 'question_h5' not in kwargs:
            raise ValueError('Must give question_h5')
        if 'feature_h5' not in kwargs:
            raise ValueError('Must give feature_h5')

        feature_h5_path = str(kwargs.pop('feature_h5'))
        print('Reading features from ', feature_h5_path)
        self.feature_h5 = h5py.File(feature_h5_path, 'r')['features']

        self.image_h5 = None
        if 'image_h5' in kwargs:
            image_h5_path = str(kwargs.pop('image_h5'))
            print('Reading images from ', image_h5_path)
            self.image_h5 = h5py.File(image_h5_path, 'r')['images']

        indices = None
        if 'indices' in kwargs:
            indices = kwargs.pop('indices')

        if 'shuffle' not in kwargs:
            # be nice, and make sure the user knows they aren't shuffling
            warnings.warn('\n\n\tYou have not provided a \'shuffle\' argument to the data loader.\n'
                      '\tBe aware that the default behavior is to NOT shuffle the data.\n')

        question_h5_path = str(kwargs.pop('question_h5'))
        with h5py.File(question_h5_path) as question_h5:
            questions = question_h5['questions']
            image_indices = question_h5['image_idxs']
            programs = question_h5['programs']
            answers = question_h5['answers']
            self.dataset = ClevrDataset(questions, image_indices, programs, self.feature_h5,
                                        answers, self.image_h5, indices)
        kwargs['collate_fn'] = clevr_collate
        super().__init__(self.dataset, **kwargs)

    def close(self):
        # Close our files to prevent leaks
        if self.image_h5 is not None:
            self.image_h5.close()
        if self.feature_h5 is not None:
            self.feature_h5.close()
        return

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
