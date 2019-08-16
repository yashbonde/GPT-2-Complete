import argparse
import logging
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import sentencepiece as spm
from model import model
from utils import sample_sequence

# parse arguments
parse = argparse.ArgumentParser(description = 'Script to train GPT2 model on any data')
parse.add_argument('--file', default = './working/nips_clean.txt', help = 'path to cleaned data txt file')
parse.add_argument('--spm_model', default='./working/nips_model.model', help='name of sentencepiece model')
parse.add_argument('--network_name', default = 'blowfish', type = str, help = 'name to give to network')
# I'm a args.network_name, bitch!
parse.add_argument('--n_ctx', default = 500, type = int, help = 'size of window (sentence) to use')
parse.add_argument('--e_dim', default = 200, type = int, help = 'embedding dimension value')
parse.add_argument('--n_stacks', default = 4, type = int, help = 'number of stacks to use')
parse.add_argument('--n_heads', default = 4, type = int, help = 'number of heads in multihead attention')
parse.add_argument('--bs', default = 40, type = int, help = 'minibatch size')
parse.add_argument('--lr', default = 0.001, type = float, help = 'learning rate')
parse.add_argument('--temprature', default = 0.7, type = float, help = 'temprature plays with variations')
parse.add_argument('--top_k', default = 40, type = int, help = 'number of samples to take in multinomial '+\
    'distributions. Basically we go over these samples [for each batch] and train the model, ' +\
    'learns by variating on the data.')
parse.add_argument('--epochs', default = 400, type = int, help = 'number of training epochs')
args = parse.parse_args()


# load data from arxiv
data_ = open(args.file, encoding = 'utf-8').readlines()
sp_model = spm.SentencePieceProcessor()
sp_model.load(args.sp_model)
vocab_size = len(open(args.sp_model.replace('.model', '.vocab'), encoding = 'utf-8').readlines())

class DataSampler:
    def __init__(self, data, encoder, seed = 4):
        all_ = []
        logging.warning('Creating Data Sampler ...')
        for i in tqdm(range(len(data))):
            d = data[i]
            all_.append([encoder.bos_id()] + encoder.encode_as_ids(d) + [encoder.eos_id()])
            
        self.total_size = sum(len(chunk) for chunk in all_)
        self.boundaries = [0]
        for i in range(len(all_)):
            self.boundaries.append(self.boundaries[-1] + len(all_[i]))
        self.rs = np.random.RandomState(seed=seed)
        self.data = all_
        
    def binary_search(self, f, lo, hi):
        if f(lo) or not f(hi):
            return None
        while hi > lo + 1:
            mid = (lo + hi) // 2
            if f(mid):
                hi = mid
            else:
                lo = mid
        return hi
        
    def sample(self, length):
        assert length < self.total_size // len(
            self.data
        ), "Dataset files are too small to sample {} tokens at a time".format(
            length)
        while True:
            index = self.rs.randint(0, self.total_size - length - 1)
            i = self.binary_search(lambda j: self.boundaries[j] > index, 0,
                              len(self.boundaries) - 1) - 1
            if self.boundaries[i + 1] > index + length:
                within_chunk = index - self.boundaries[i]
                return self.data[i][within_chunk:within_chunk + length]
            
ds = DataSampler(data_, sp_model)
# print('*** sample data: {}'.format(ds.sample(40)))

class ModelConfig:
    def __init__(self):
        pass

    def add_config(self, name, value):
        setattr(self, name, value)

# make configuration
config = ModelConfig()
config.add_config('num_context', args.n_ctx)
config.add_config('embedding_dim', args.e_dim)
config.add_config('vocab_size', vocab_size)
config.add_config('num_layers', args.n_stacks)
config.add_config('num_heads', args.n_heads)
config.add_config('batch_size', args.bs)
config.add_config('learning_rate', args.lr)
config.add_config('temprature', args.temprature)
config.add_config('top_k', args.top_k)
config.add_config('num_epochs', args.epochs)

# constants
NAME = args.network_name
tb_path = './' + NAME
model_save_path = tb_path + '/{}.ckpt'.format(NAME)
sample_save_path = tb_path + '/dumps.txt'

# print([i for i in dir(config) if '__' not in i])

def randomize(context, config, p):
    if p > 0:
        mask = tf.random.uniform(shape = tf.shape(context))
        noise = tf.random.uniform(shape = tf.shape(context),
                                  minval = 0,
                                  maxval = config.vocab_size,
                                  dtype =tf.int32)
        return tf.where(mask, noise, context)
    return context

# make tf placeholders
context = tf.placeholder(tf.int32, [config.batch_size, None], name = 'context_placeholder')
context_in = randomize(context, config, 0)
print('** context: {}\n** context_in: {}'.format(context, context_in))

output = model(config = config,
               inp = context_in)

# print
print('** output logits:', output['logits'])
print('** output present:', output['present'])

# print stats
p_ = tf.unstack(output['present'], axis = 1)
for i in range(len(p_)):
    print('%% Output of stack {}: {}'.format(i, p_[i]))

# make loss
loss = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels = context[:, 1:],
        logits = output['logits'][:, :-1]
    )
)
tf.summary.scalar('loss', loss)
print('** loss:', loss)

# sample sequence
tf_sample = sample_sequence(config = config,
                            length = config.num_context,
                            start_token = sp_model.bos_id(),
                            temprature = config.temprature,
                            top_k = config.top_k)
print('** tf_sample:', tf_sample)

# train vars and train steps
train_vars = [v for v in tf.trainable_variables() if 'model' in v.name]
opt = tf.train.AdamOptimizer(config.learning_rate)
train_step = opt.minimize(loss, var_list=train_vars)

# saver and file writer
saver_ = tf.train.Saver(max_to_keep = 2)

# encoder function
def encode(string):
    x = sp_model.encode_as_ids(string)
    x = x[:config.num_context]
    '''
    for _ in range(config.num_context - len(x)):
        x.append(sp_model.pad_id())
    '''
    return x

# decoder function
def decode(pieces):
    x = sp_model.decode_ids(pieces.tolist())
    return x

# save function
def save_network(sess):
    saver_.save(sess, model_save_path)

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(tb_path, sess.graph)
    sess.run(tf.global_variables_initializer())
    merged_summary = tf.summary.merge_all()
    global_step = 0

    for epoch_idx in range(config.num_epochs):
        print('======== EPOCH {} ========'.format(epoch_idx))
        for j in tqdm(range(0, len(data_), len(data_) // config.batch_size)):
            samp_ = [ds.sample(config.num_context) for _ in range(config.batch_size)]
            if len(samp_) != config.batch_size:
                continue

            _, l, sum_ = sess.run([train_step, loss, merged_summary], feed_dict = {context: samp_})

            train_writer.add_summary(sum_, global_step)
            global_step += 1

        # dump every epoch
        print('Adding to dump ... (might take some time)')
        t_ = '\n======== EPOCH {} ========\n'.format(epoch_idx)
        s_ = sess.run(tf_sample).astype(np.int32)
        for ss in s_:
            t_ += decode(ss) + '\n'
        with open(sample_save_path, 'a', encoding = 'utf-8') as f:
            f.write(t_)

        if epoch_idx % 2 == 0:
            save_network(sess)
