"""
gpt.utils.py

28.09.2019 - @yashbonde
"""

import json
import os
from functools import lru_cache
from glob import glob
from tqdm import tqdm
import regex as re
import tensorflow as tf
import numpy as np

from model import model, past_shape

'''
accumulator.py
'''

class AccumulatingOptimizer(object):
    def __init__(self, opt, var_list):
        self.opt = opt
        self.var_list = var_list
        self.accum_vars = {tv : tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False)
                           for tv in var_list}
        self.total_loss = tf.Variable(tf.zeros(shape=[], dtype=tf.float32))
        self.count_loss = tf.Variable(tf.zeros(shape=[], dtype=tf.float32))

    def reset(self):
        updates = [tv.assign(tf.zeros_like(tv)) for tv in self.accum_vars.values()]
        updates.append(self.total_loss.assign(tf.zeros(shape=[], dtype=tf.float32)))
        updates.append(self.count_loss.assign(tf.zeros(shape=[], dtype=tf.float32)))
        with tf.control_dependencies(updates):
            return tf.no_op()

    def compute_gradients(self, loss):
        grads = self.opt.compute_gradients(loss, self.var_list)
        updates = [self.accum_vars[v].assign_add(g) for (g,v) in grads]
        updates.append(self.total_loss.assign_add(loss))
        updates.append(self.count_loss.assign_add(1.0))
        with tf.control_dependencies(updates):
            return tf.no_op()

    def apply_gradients(self):
        grads = [(g,v) for (v,g) in self.accum_vars.items()]
        with tf.control_dependencies([self.opt.apply_gradients(grads)]):
            return self.total_loss / self.count_loss


'''
sample.py --> functions from sample.py file
'''

def top_k_logits(logits, k):
    """
    return a tensor of top k logits from input logits
    """
    if k == 0:
        # no truncation
        return logits

    def top_k():
        values, _ = tf.nn.top_k(logits, k=k)
        min_values = values[:, -1, tf.newaxis]
        out = tf.where(
            logits < min_values,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits
        )
        return out

    out = tf.cond(
        tf.equal(k, 0),
        lambda: logits,
        lambda: top_k(),
    )
    return out

def top_p_logits(logits, p):
    with tf.variable_scope('top_p_logits'):
        logits_sort = tf.sort(logits, direction = 'DESCENDING')
        probs_sort = tf.nn.softmax(logits_sort)
        probs_sum = tf.cumsum(probs_sort, axis = 1, exclusive=True)

        # [batch_size, vocab]
        logits_masked = tf.where(probs_sum < p, logits_sort, tf.ones_like(logits_sort) * 1000)
        # [batch_size, 1]
        min_logits = tf.reduce_mim(logits_masked, axis = 1, keepdims = True)
        return tf.where(
            logits < min_logits,
            tf.ones_like(logits, dtype = logits.dtype) * -1e10,
            logits,
        )


def sample_sequence(config, length, start_token = None, context = None, temprature = 1, top_k = 0,
                     top_p = 0.0):
    if start_token is None:
        assert context is not None, 'Specify either `start_token` or `context`'
    else:
        assert context is None, 'Specify either `start_token` or `context`'
        context = tf.fill([config.batch_size, 1], start_token)

    def step(config, tokens, past = None):
        lm_output = model(config, tokens, past, reuse = tf.AUTO_REUSE)

        logits = lm_output['logits'][:, :, :config.vocab_size]
        present = lm_output['present']
        present.set_shape(past_shape(config))
        return {
            'logits': logits,
            'present': present
        }

    with tf.name_scope('sample_sequence'):
        # Don't feed the last context token -- leave that to the loop below
        '''TODO: Would be slightly faster if we called step on the entire context
        rather than leaving the last token transformer calculation to the while loop'''
        context_output = step(config, context[:, :-1])

        def body(past, prev, output):
            next_outputs = step(config, prev[:, tf.newaxis], past)
            logits = next_outputs['logits'][:, -1, :] / tf.cast(temprature, tf.float32)
            if top_p > 0.0:
                logits = top_p_logits(logits, top_p)
            else:
                logits = top_k_logits(logits, top_k)

            samples = tf.multinomial(logits, num_samples = 1, output_dtype=tf.int32)
            return [
                tf.concat([past, next_outputs['present']], axis = -2),
                tf.squeeze(samples, axis = [1]),
                tf.concat([output, samples], axis = 1)
            ]

        def cond(t1, t2, t3):
            return True


        _, _, tokens = tf.while_loop(
            cond = cond,
            body = body,
            maximum_iterations = length,
            loop_vars = [
                context_output['present'],
                context[:, -1],
                context
            ],
            shape_invariants = [
                tf.TensorShape(past_shape(config)),
                tf.TensorShape([config.batch_size]),
                tf.TensorShape([config.batch_size, None])
            ],
            back_prop = False
        )

        return tokens

'''
encoder.py --> All functions from encoder.py are below
'''

@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and corresponding list of unicode strings. The reversible bpe codes
    work on unicode strings. This means you need a large number of unicode characters in your
    vocab if you want to avoid UNKs. When you're at something like 10B token dataset you end up
    needing around 5K for a decent converage. This is significant percentage of your normal, say
    32K bpe vocab. To avoid that we want lookup tables between utf-8 bytes and unicode strings.
    And avoind mapping to whitespace/control characters and bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + \
         list(range(ord("¡"), ord("¬") + 1)) + \
         list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    out = dict(zip(bs, cs))
    return out


def get_pairs(word):
    """
    get character-level bigrams of input word
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class Encoder:
    def __init__(self, encoder, bpe_merges, errors='replace'):
        self.encoder = encoder
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break

            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])

            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break

            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
        return text


def get_encoder(model_name, models_dir):
    with open(os.path.join(models_dir, model_name, 'encoder.json'), 'r') as f:
        encoder = json.load(f)
    with open(os.path.join(models_dir, model_name, 'vocab.bpe'), 'r', encoding='utf-8') as f:
        bpe_data = f.read()

    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1: -1]]
    return Encoder(
        encoder=encoder,
        bpe_merges=bpe_merges
    )

'''
load_dataset.py --> All functions from load_dataset.py are here
'''

def load_dataset(enc, path, combine):
    paths = []
    if os.path.isfile(path):
        # simple file
        paths.append(path)
    elif os.path.isdir(path):
        for dirpath, _, fnames in os.walk(path):
            for fname in fnames:
                paths.append(os.path.join(dirpath, fname))

    else:
        # assume glob
        paths = glob(path)

    token_chunks = []
    raw_text = ''
    for path in tqdm(paths):
        if path.endswith('.npz'):
            # pre-encoded
            with np.load(path) as npz:
                for item in npz.files:
                    token_chunks.append(npz[item])
        else:
            # plain text
            with open(path, 'r') as fp:
                raw_text += fp.read()
            if len(raw_text) >= combine:
                tokens = np.stack(enc.encode(raw_text))
                token_chunks.append(tokens)
                raw_text = ''
            else:
                raw_text += '<|endoftext|>'

    if raw_text:
        tokens = np.stack(enc.encode(raw_text))
        token_chunks.append(tokens)

    return token_chunks

def binary_search(f, lo, hi):
    if f(lo) or not f(hi):
        return None
    while hi > lo + 1:
        mid = (lo + hi) // 2
        if f(mid):
            hi = mid
        else:
            lo = mid
    return hi

class Sampler:
    """
    Fairly samples a slice from a set of variable sized chunks
    'Fairly' means that the distribution is same as sampling from one
    concatenated chunk, but without crossing chunk boundaries.
    """
    def __init__(self, chunks, seed = None):
        self.chunks = chunks
        self.total_size = sum(chunk.shape[0] for chunk in chunks)
        self.boundaries = [0]
        for i in range(len(chunks)):
            self.boundaries.append(self.boundaries[-1] + chunks[i].shape[0])
        self.rs = np.random.RandomState(seed = seed)

    def sample(self, length):
        assert length < self.total_size // len(self.chunks) , \
            "Dataset files are too small to sample {} tokens at a time".format(length)

        while True:
            index = self.rs.randint(0, self.total_size - length - 1)
            i = binary_search(lambda  j: self.boundaries[i] > index, 0,
                              len(self.boundaries) - 1) - 1
            if self.boundaries[i+1] > index + length:
                within_chunk = index - self.boundaries[i]
                return self.chunks[i][within_chunk: within_chunk + length]

