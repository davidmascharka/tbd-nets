# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license available at
# https://github.com/facebookresearch/clevr-iep/blob/master/LICENSE
#
# Modifications by David Mascharka to update the code for compatibility with PyTorch >0.1 lead to:
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
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import h5py
from pathlib import Path

__all__ = ['load_model', 'generate_programs']

class _Seq2Seq(nn.Module):
    def __init__(self, 
        encoder_vocab_size=100,
        decoder_vocab_size=100,
        wordvec_dim=300,
        hidden_dim=256,
        rnn_num_layers=2,
        rnn_dropout=0,
        null_token=0,
        start_token=1,
        end_token=2,
        encoder_embed=None):
        super().__init__()
        self.encoder_embed = nn.Embedding(encoder_vocab_size, wordvec_dim)
        self.encoder_rnn = nn.LSTM(wordvec_dim, hidden_dim, rnn_num_layers,
                                   dropout=rnn_dropout, batch_first=True)
        self.decoder_embed = nn.Embedding(decoder_vocab_size, wordvec_dim)
        self.decoder_rnn = nn.LSTM(wordvec_dim + hidden_dim, hidden_dim, rnn_num_layers,
                                   dropout=rnn_dropout, batch_first=True)
        self.decoder_linear = nn.Linear(hidden_dim, decoder_vocab_size)
        self.NULL = null_token
        self.START = start_token
        self.END = end_token

    def get_dims(self, x=None, y=None):
        V_in = self.encoder_embed.num_embeddings
        V_out = self.decoder_embed.num_embeddings
        D = self.encoder_embed.embedding_dim
        H = self.encoder_rnn.hidden_size
        L = self.encoder_rnn.num_layers

        N = x.size(0) if x is not None else None
        N = y.size(0) if N is None and y is not None else N
        T_in = x.size(1) if x is not None else None
        T_out = y.size(1) if y is not None else None
        return V_in, V_out, D, H, L, N, T_in, T_out

    def before_rnn(self, x, replace=0):
        N, T = x.size()
        idx = torch.LongTensor(N).fill_(T - 1)
        x_cpu = x.cpu()
        for i in range(N):
            for t in range(T - 1):
                if x_cpu[i, t] != self.NULL and x_cpu[i, t + 1] == self.NULL:
                    idx[i] = t
                    break
        idx = idx.type_as(x)
        x[x == self.NULL] = replace
        return x, idx

    def encoder(self, x):
        V_in, V_out, D, H, L, N, T_in, T_out = self.get_dims(x=x)
        x, idx = self.before_rnn(x)
        embed = self.encoder_embed(x)
        h0 = torch.zeros(L, N, H).type_as(embed)
        c0 = torch.zeros(L, N, H).type_as(embed)
        out, _ = self.encoder_rnn(embed, (h0, c0))
        idx = idx.view(N, 1, 1).expand(N, 1, H)
        return out.gather(1, idx).view(N, H)

    def decoder(self, encoded, y, h0=None, c0=None):
        V_in, V_out, D, H, L, N, T_in, T_out = self.get_dims(y=y)

        if T_out > 1:
            y, _ = self.before_rnn(y)
        y_embed = self.decoder_embed(y)
        encoded_repeat = encoded.view(N, 1, H)
        encoded_repeat = encoded_repeat.expand(N, T_out, H)
        rnn_input = torch.cat([encoded_repeat, y_embed], 2)
        if h0 is None:
            h0 = torch.zeros(L, N, H).type_as(encoded)
        if c0 is None:
            c0 = torch.zeros(L, N, H).type_as(encoded)
        rnn_output, (ht, ct) = self.decoder_rnn(rnn_input, (h0, c0))

        rnn_output_2d = rnn_output.contiguous().view(N * T_out, H)
        output_logprobs = self.decoder_linear(rnn_output_2d).view(N, T_out, V_out)

        return output_logprobs, ht, ct

    def reinforce_sample(self, x, max_length=30):
        N, T = x.size(0), max_length
        encoded = self.encoder(x)
        y = torch.LongTensor(N, T).fill_(self.NULL)
        done = torch.ByteTensor(N).fill_(0)
        cur_input = x.new(N, 1).fill_(self.START)
        h, c = None, None
        for t in range(T):
            # logprobs is N x 1 x V
            logprobs, h, c = self.decoder(encoded, cur_input, h0=h, c0=c)
            probs = F.softmax(logprobs.view(N, -1), dim=1) # Now N x V
            _, cur_output = probs.max(1)
            cur_output = cur_output.unsqueeze(1)
            cur_output_data = cur_output.cpu()
            not_done = logical_not(done)
            y[:, t][not_done] = cur_output_data[not_done][0]
            done = logical_or(done, cur_output_data.cpu().squeeze() == self.END)
            cur_input = cur_output
            if done.sum() == N:
                break
        return y.type_as(x)

def logical_or(x, y):
    return (x + y).clamp_(0, 1)

def logical_not(x):
    return x == 0

def load_program_generator(checkpoint):
    """ Loads the program generator model from `checkpoint`.

    Parameters
    ----------
    checkpoint : Union[pathlib.Path, str]
        The path to a checkpoint file.

    Returns
    -------
    torch.nn.Module
        The program generator model, which takes as input a question and produces a logical series
        of operations that can be used to answer that question.
    """
    checkpoint = torch.load(str(checkpoint), map_location={'cuda:0': 'cpu'})
    kwargs = checkpoint['program_generator_kwargs']
    state = checkpoint['program_generator_state']
    program_generator = _Seq2Seq(**kwargs)
    program_generator.load_state_dict(state)
    return program_generator

def generate_single_program(question, program_generator, vocab, question_len=46):
    """ Generate a single program from a given natural-language question using the provided model.
    
    Parameters
    ----------
    question : str
        The question to produce a program from.

    program_generator : torch.nn.Module
        The program generation model to use to produce a program.

    vocab : Dict[str, Dict[any, any]]
        The dictionary to use to convert words to indices.

    Returns
    -------
    torch.Tensor
        The program encoding the logical steps to perform in answering the question.
    """
    # remove punctuation from our question
    import re
    punc = '!"#$%&\'()*+-./:<=>?@[\\]^_`{|}~' # string.punctuation excluding comma and semicolon
    punctuation_regex = re.compile('[{}]'.format(re.escape(punc)))
    question = punctuation_regex.sub('', question).split()

    # tell the user they can't use unknown words
    question_token_to_idx = vocab['question_token_to_idx']
    if any(word not in question_token_to_idx for word in question):
        print('Error: there are unknown words in the question you provided!')
        print('Unknown words:')
        print([word for word in question if word not in question_token_to_idx])
        assert False

    # encode the question using our vocab
    encoded = np.zeros((1, question_len), dtype='int64')
    encoded[0, 0] = question_token_to_idx['<START>']
    encoded[0, 1:len(question)+1] = [question_token_to_idx[word] for word in question]
    encoded[0, len(question)+1] = question_token_to_idx['<END>']

    question_tensor = torch.LongTensor(encoded)

    # push to the GPU if we can
    if torch.cuda.is_available():
        program_generator.cuda()
        question_tensor = question_tensor.cuda()
    program_generator.eval()
    
    # generate a program
    return program_generator.reinforce_sample(question_tensor)
    

def generate_programs(h5_file, program_generator, dest_dir, batch_size):
    """ Generate programs from a given HDF5 file containing questions.

    Parameters
    ----------
    h5_file : Union[pathlib.Path, str]
        Path to hdf5 file containing the questions and image-indices.

    program_generator : torch.nn.Module
        The program generation model to use to produce programs.

    dest_dir : Union[pathlib.Path, str]
        Path to store the output program, image index, and question .npy files.

    batch_size : Integral
        How many programs to process at once.

    Returns
    -------
    None
    """
    with h5py.File(str(h5_file)) as questions_h5:
        questions = np.asarray(questions_h5['questions'])
        image_indices = np.asarray(questions_h5['image_idxs'])

        if torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor

        program_generator.type(dtype).eval()

        print('Generating programs...')
        progs = []
        for start_idx in range(0, len(questions), batch_size):
            question_batch = questions[start_idx:start_idx+batch_size]
            questions_var = torch.LongTensor(question_batch).type(dtype).long()
            for question in questions_var:
                program = program_generator.reinforce_sample(question.view(1, -1))
                progs.append(program.cpu().numpy().squeeze())
        progs = np.asarray(progs)
    
    dest = Path(dest_dir)
    path = dest / 'programs.npy'
    np.save(path, progs)
    print('Saved programs as {}'.format(path.absolute()))
    path = dest / 'image_idxs.npy'
    np.save(path, image_indices)
    print('Saved image indices as {}'.format(path.absolute()))
    path = dest / 'questions.npy'
    np.save(path, questions)
    print('Saved questions as {}'.format(path.absolute()))
