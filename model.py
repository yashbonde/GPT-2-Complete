"""
tf_layers.py

Layer functions for the GPT-2 based model

27.09.2019 - @yashbonde
"""

import numpy as np
import tensorflow as tf


def gelu_activation(inp):
    """
    Gaussian Error Linear Unit (GELU) is a new type of activation function that can
    estimate any of the existing activation values such as Sigmoid, ReLU, ELU, tanh
    while providing superior learning.
    See this [paper](https://arxiv.org/pdf/1606.08415.pdf)

    :param inp: input tensor
    :return:
    """
    out = 1 + tf.tanh(np.sqrt(np.pi) * (inp + 0.044715 * tf.pow(inp, 3)))
    out *= 0.5 * inp
    return out


def shapes_list(inp):
    """
    cleaner handling of tensorflow shapes
    :param inp: input tensor
    :return: list of shapes combining dynamic and static shapes
    """
    shapes_static = inp.get_shape().as_list()
    shapes_dynamic = tf.shape(inp)
    cleaned_shape = [shapes_dynamic[i] if s is None else s for i, s in enumerate(shapes_static)]
    return cleaned_shape


def softmax_with_reduce_max(inp, axis=-1):
    """
    perform softmax, this is slightly different to the default softmax in tensorflow
    :param inp:
    :param axis:
    :return:
    """
    out = inp - tf.reduce_max(inp, axis=axis, keepdims=True)
    ex = tf.exp(out)
    sm = ex / tf.reduce_sum(ex, axis=axis, keepdims=True)
    return sm


def normalise_tensor(inp, scope, *, axis=-1, epsilon=1e-5):
    """
    Normalize the input values between 0 and 1, then do diagonal affine transform
    :param inp: input tensor
    :param scope: tf variable scope
    :param axis: axis to perform ops on
    :param epsilon: base minimum value
    :return: normalised tensor
    """
    with tf.variable_scope(scope):
        e_dim = inp.get_shape().as_list()[-1]
        g = tf.get_variable('g', [e_dim], initializer=tf.constant_initializer(1))
        b = tf.get_variable('b', [e_dim], initializer=tf.constant_initializer(0))

        u = tf.reduce_mean(inp, axis=axis, keepdims=True)
        s = tf.reduce_mean(tf.square(inp - u), axis=axis, keepdims=True)
        out = (inp - u) * tf.rsqrt(s + epsilon)
        out = out * g + b

        return out


def split_into_n_states(inp, n):
    """2
    reshape last dimension of input tensor from n --> [n, inp.shape[-1]/n]
    :param inp: input tensor
    :param n: number of splits
    :return: reshaped tensor
    """
    *start, m = shapes_list(inp)
    out = tf.reshape(inp, start + [n, m // n])
    return out


def merge_n_states(inp):
    """
    merge the last two dimensions
    :param inp: input tensor
    :return: reshaped tensor
    """
    *start, m, n = shapes_list(inp)
    out = tf.reshape(inp, start + [m * n])
    return out


def conv1d(inp, scope, num_features, weights_init_stddev=0.2):
    """
    1D convolutional block, first reshape input then matmul weights and then reshape

    :param inp: input tensor
    :param scope: tf variable scope
    :param num_features: number of output features
    :param weights_init_stddev: standard deviation value
    :return: processed output
    """
    with tf.variable_scope(scope):
        *start, nx = shapes_list(inp)
        weights = tf.get_variable('w', [1, nx, num_features],
                                  initializer=tf.random_normal_initializer(stddev=weights_init_stddev))
        bias = tf.get_variable('b', [num_features],
                               initializer=tf.constant_initializer(0))

        # reshape input and weights and perform matmul and add bias
        inp_reshaped = tf.reshape(inp, [-1, nx])
        w_reshaped = tf.reshape(weights, [-1, num_features])
        out = tf.matmul(inp_reshaped, w_reshaped) + bias

        out = tf.reshape(out, start + [num_features])
        return out


def attention_mask(nd, ns, dtype=tf.float32):
    """
    1's in the lower traingle, couting from lower right corner

    This is same as using the tf.matrix_band_part() but it doesn't produce garbage on TPUs

    :param nd:
    :param ns:
    :param dtype:
    :return:
    """
    i = tf.range(nd)[:, None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    out = tf.cast(m, dtype)
    return out


def attention(inp, scope, e_dim, past, config):
    """
    complete attention model in a single function

    :param inp: input tensor
    :param scope: tf variable scope
    :param e_dim: embedding dimension value
    :param past: previous outputs ??
    :param config: config file
    :return: attention value and present value
    """
    assert inp.shape.ndims == 3  # input should be of shape [batch, seqlen, embeddings] # [batch, sequence, features]
    assert e_dim % config.num_heads == 0  # embedding can be split in heads

    if past is not None:
        assert past.shape.ndims == 5  # [batch, 2, heads, seqlen, emebeddings]

    def split_heads(x):
        out = split_into_n_states(x, config.num_heads)
        out = tf.transpose(out, [0, 2, 1, 3])
        return out

    def merge_heads(x):
        out = merge_n_states(tf.transpose(x, [0, 2, 1, 3]))
        return out

    def mask_attention_weights(w):
        # w should have shape [batches, heads, dst_seq, src_seq], where information flows from scr to dst
        _, _, nd, ns = shapes_list(w)
        b = attention_mask(nd, ns, w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w * b - tf.cast(1e10, w.dtype) * (1 - b)
        return w

    def multihead_attention(q, k, v):
        w = tf.matmul(q, k, transpose_b=True)
        w *= tf.rsqrt(tf.cast(v.shape[-1].value, w.dtype))

        # mask attention weights
        w = mask_attention_weights(w)
        w = softmax_with_reduce_max(w)
        out = tf.matmul(w, v)
        return out

    with tf.variable_scope(scope):
        c = conv1d(inp, 'convolutional_attention', e_dim * 3)
        q, k, v = map(split_heads, tf.split(c, 3, axis=2))
        present = tf.stack([k, v], axis=1)
        if past is not None:
            # there is a stack below it
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk, k], axis=2)
            v = tf.concat([pv, v], axis=2)

        attn = multihead_attention(q, k, v)
        attn = merge_heads(attn)

        out = conv1d(attn, 'convolutional_projection', e_dim)
        return out, present


def multilayer_perceptron(inp, scope, hidden_dim):
    """
    MLP

    :param inp: input tensor
    :param scope: tf variable scope
    :param hidden_dim: hidden dimension
    :return: output processed tensor
    """
    with tf.variable_scope(scope):
        nx = inp.shape[-1].value
        out = conv1d(inp, 'convolutional_ff', hidden_dim)
        out = gelu_activation(out)
        out = conv1d(out, 'convolutional_projection', nx)
        return out


def block(inp, scope, past, config):
    """
    one stack or block with multihead attention and ff block

    :param inp: input tensor
    :param scope: tf variable scope
    :param past: past tensors
    :param config: config object
    :return: processed output and
    """
    with tf.variable_scope(scope):
        nx = inp.shape[-1].value
        norm = normalise_tensor(inp, 'ln_1')
        attn, present = attention(norm, 'attn', nx, past=past, config=config)
        out = attn + inp
        norm = normalise_tensor(out, 'ln_2')
        mlp_out = multilayer_perceptron(norm, 'mlp', nx * 4)  # note that hidden dim is 4x
        out = out + mlp_out
        return out, present


def past_shape(config, seqlen=None):
    """
    return a list with shape of `past` tensor

    :param config: config object
    :return: list with shape value
    """
    shape = [config.batch_size, config.num_layers, 2, config.num_heads, seqlen,
             config.embedding_dim // config.num_heads]
    return shape


def expand_tile(value, size):
    """
    expand value to size

    :param value: input object to be tiles
    :param size: size to tile the object to
    :return: tiled output
    """
    value = tf.convert_to_tensor(value, name='value')
    ndims = value.shape.ndims
    out = tf.expand_dims(value, axis=0)
    out = tf.tile(out, [size] + [1, ] * ndims)
    return out


def positions_for(tokens, past_length):
    """
    get positions only for a input tokens

    :param tokens: input tokens
    :param past_length: length of past object
    :return: output
    """
    batch_size = tf.shape(tokens)[0]
    nsteps = tf.shape(tokens)[1]
    out = expand_tile(past_length + tf.range(nsteps), batch_size)
    return out


def model(config, inp, past=None, scope='model', reuse=False):
    """
    Model function which returns one complete model

    :param config: ModelConfig file
    :param inp: input tensor for generation
    :param past: any past tensors
    :param scope: scope of the model
    :param reuse: to reuse the model
    :return: dictionary with two objects
    """
    with tf.variable_scope(scope, reuse=reuse):
        results = {}
        batch_size, seqlen = shapes_list(inp)

        wpe = tf.get_variable('wpe', [config.num_context, config.embedding_dim],
                              initializer=tf.random_normal_initializer(stddev=0.01))
        wte = tf.get_variable('wte', [config.vocab_size, config.embedding_dim],
                              initializer=tf.random_normal_initializer(stddev=0.02))

        past_length = 0 if past is None else tf.shape(past)[-2]
        h = tf.gather(wte, inp) + tf.gather(wpe, positions_for(inp, past_length))

        # Transformer
        presents = []  # all the layer outputs
        pasts = tf.unstack(past, axis=1) if past is not None else [None, ] * config.num_layers
        assert len(pasts) == config.num_layers

        for layer, past in enumerate(pasts):
            h, present = block(h, 'stack_{}'.format(layer), past=past, config=config)
            presents.append(present)

        results['present'] = tf.stack(presents, axis=1)
        out = normalise_tensor(h, 'ln_f')

        # language model loss
        h_flat = tf.reshape(out, [batch_size * seqlen, config.embedding_dim])
        logits = tf.matmul(h_flat, wte, transpose_b=True)
        logits = tf.reshape(logits, [batch_size, seqlen, config.vocab_size], name='logits')
        results['logits'] = logits

        return results
