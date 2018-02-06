# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Basic word2vec example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import math
import os
import random
import csv
import configargparse
import zipfile
from copy import deepcopy

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


class Word2Vec():
    def __init__(self, config):
        self.config = config
        self.data_index = 0

    # Read the data into a list of strings.
    def read_data(self):
        """Extract the first file enclosed in a zip file as a list of words."""
        filename = self.config.train_file
        with open(filename, 'r') as f:
            data = tf.compat.as_str(f.read()).split()
        return data

    def build_dataset(self, words, n_words):
        """Process raw inputs into a dataset."""
        count = [['UNK', -1]]
        count.extend(collections.Counter(words).most_common(n_words - 1))
        vocab_size = len(count)
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
        data = list()
        unk_count = 0
        for word in words:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0  # dictionary['UNK']
                unk_count += 1
            data.append(index)
        count[0][1] = unk_count
        reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return data, count, dictionary, reversed_dictionary, vocab_size

    def save_token_and_count(self, count, reverse_dictionary):
        token_index_file = self.config.token_index_file
        count_of_words_file = self.config.count_of_words_file

        with open(token_index_file, 'w') as f:
            writer = csv.writer(f)
            for key, val in reverse_dictionary.items():
                writer.writerow([key, val])
        with open(count_of_words_file, 'w') as f:
            writer = csv.writer(f)
            for row in count:
                writer.writerow(row)

    # Step 3: Function to generate a training batch for the skip-gram model.
    def generate_batch(self, data, batch_size, num_skips, skip_window):
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * skip_window + 1  # [ skip_window target skip_window ]
        buffer = collections.deque(maxlen=span)
        for _ in range(span):
            buffer.append(data[self.data_index])
            self.data_index = (self.data_index + 1) % len(data)
        for i in range(batch_size // num_skips):
            target = skip_window  # target label at the center of the buffer
            targets_to_avoid = [skip_window]
            for j in range(num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i * num_skips + j] = buffer[skip_window]
                labels[i * num_skips + j, 0] = buffer[target]
            buffer.append(data[self.data_index])
            self.data_index = (self.data_index + 1) % len(data)
        # Backtrack a little bit to avoid skipping words in the end of a batch
        self.data_index = (self.data_index + len(data) - span) % len(data)
        return batch, labels

    # Step 6: Visualize the embeddings.
    def plot_with_labels(self, final_embeddings, reverse_dictionary, filename='tsne.png'):
        try:
            # pylint: disable=g-import-not-at-top
            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt

            tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
            plot_only = min(500, len(reverse_dictionary))
            low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
            labels = [reverse_dictionary[i] for i in xrange(plot_only)]

            assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
            plt.figure(figsize=(18, 18))  # in inches
            for i, label in enumerate(labels):
                x, y = low_dim_embs[i, :]
                plt.scatter(x, y)
                plt.annotate(label,
                             xy=(x, y),
                             xytext=(5, 2),
                             textcoords='offset points',
                             ha='right',
                             va='bottom')

            plt.savefig(filename)

        except ImportError:
            print('Please install sklearn, matplotlib, and scipy to show embeddings.')

    def save_embeddings(self, file, embeddings, reverse_dictionary):
        labels = reverse_dictionary.values()
        with open(file, "w") as f:
            for i, label in enumerate(labels):
                vec = embeddings[i, :]
                str_vec = map(str, vec)
                # print("{label} {embeddings}".format(label=label, embeddings=" ".join(str_vec)))
                f.write("{label} {embeddings}\n".format(label=label, embeddings=" ".join(str_vec)))

    def build_and_run_model(self, data, reverse_dictionary):
        config = self.config

        batch_size = config.batch_size
        embedding_size = config.embedding_size
        skip_window = config.skip_window
        num_skips = config.num_skips
        vocabulary_size = config.vocabulary_size
        valid_size = config.valid_size
        valid_window = vocabulary_size
        num_steps = config.num_steps
        ckpt_save_path = config.ckpt_save_path

        # Step 4: Build and train a skip-gram model.
        # We pick a random validation set to sample nearest neighbors. Here we limit the
        # validation samples to the words that have a low numeric ID, which by
        # construction are also the most frequent.
        valid_examples = np.random.choice(valid_window, valid_size, replace=False)
        num_sampled = 64  # Number of negative examples to sample.
        graph = tf.Graph()
        with graph.as_default():
            # Input data.
            train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
            train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
            valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

            # Ops and variables pinned to the CPU because of missing GPU implementation
            with tf.device('/cpu:0'):
                # Look up embeddings for inputs.
                embeddings = tf.Variable(
                    tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
                embed = tf.nn.embedding_lookup(embeddings, train_inputs)

                # Construct the variables for the NCE loss
                nce_weights = tf.Variable(
                    tf.truncated_normal([vocabulary_size, embedding_size],
                                        stddev=1.0 / math.sqrt(embedding_size)))
                nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

            # Compute the average NCE loss for the batch.
            # tf.nce_loss automatically draws a new sample of the negative labels each
            # time we evaluate the loss.
            loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=nce_weights,
                               biases=nce_biases,
                               labels=train_labels,
                               inputs=embed,
                               num_sampled=num_sampled,
                               num_classes=vocabulary_size))

            # Construct the SGD optimizer using a learning rate of 1.0.
            optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

            # Compute the cosine similarity between minibatch examples and all embeddings.
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            normalized_embeddings = embeddings / norm
            valid_embeddings = tf.nn.embedding_lookup(
                normalized_embeddings, valid_dataset)
            similarity = tf.matmul(
                valid_embeddings, normalized_embeddings, transpose_b=True)

            # Add variable initializer.
            init = tf.global_variables_initializer()

        # Step 5: Begin training.
        saver = tf.train.Saver({"embeddings": embeddings})
        with tf.Session(graph=graph) as session:
            # We must initialize all variables before we use them.
            init.run()
            print('Initialized')

            average_loss = 0
            for step in xrange(num_steps):
                batch_inputs, batch_labels = self.generate_batch(
                    data, batch_size, num_skips, skip_window)
                feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

                # We perform one update step by evaluating the optimizer op (including it
                # in the list of returned values for session.run()
                _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
                average_loss += loss_val

                if step % 2000 == 0:
                    if step > 0:
                        average_loss /= 2000
                    # The average loss is an estimate of the loss over the last 2000 batches.
                    print('Average loss at step ', step, ': ', average_loss)
                    average_loss = 0

                # Note that this is expensive (~20% slowdown if computed every 500 steps)
                if step % 10000 == 0:
                    self.print_sample_similarity(reverse_dictionary, similarity, valid_examples, valid_size)
                    saved_path = saver.save(session, ckpt_save_path)
                    print("Model saved in file: %s" % saved_path)

            final_embeddings = normalized_embeddings.eval()
        return final_embeddings

    def print_sample_similarity(self, reverse_dictionary, similarity, valid_examples, valid_size):
        # return
        sim = similarity.eval()
        for i in xrange(valid_size):
            valid_word = reverse_dictionary[valid_examples[i]]
            top_k = 8  # number of nearest neighbors
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            log_str = 'Nearest to %s:' % valid_word
            for k in xrange(top_k):
                close_word = reverse_dictionary[nearest[k]]
                log_str = '%s %s,' % (log_str, close_word)
            print(log_str)

    def test_batch(self, data, reverse_dictionary):
        batch, labels = self.generate_batch(data, batch_size=8, num_skips=2, skip_window=1)
        for i in range(8):
            print(batch[i], reverse_dictionary[batch[i]],
                  '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

    def train_data(self):
        config = self.config
        vocabulary_size = config.vocabulary_size
        token_embeddings_file = config.token_embeddings_file

        vocabulary = self.read_data()
        print('Data size', len(vocabulary))

        data, count, dictionary, reverse_dictionary, correct_vocab_size = self.build_dataset(vocabulary,
                                                                                             vocabulary_size)

        if correct_vocab_size < vocabulary_size:
            print("Real vocabulary size is {real_vocab}, "
                  "while the config is {conf_vocab}. "
                  "Setting it to {real_vocab}.".format(real_vocab=correct_vocab_size, conf_vocab=vocabulary_size))
            self.config.vocabulary_size = correct_vocab_size

        del vocabulary  # Hint: to reduce memory.
        print('Most common words (+UNK)', count[:5])
        print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

        self.save_token_and_count(count, reverse_dictionary)

        self.test_batch(data, reverse_dictionary)

        final_embeddings = self.build_and_run_model(data, reverse_dictionary)

        self.plot_with_labels(final_embeddings, reverse_dictionary, os.path.join(config.output_path, 'tsne.png'))

        self.save_embeddings(token_embeddings_file, final_embeddings, reverse_dictionary)

    def test_vocab_size(self):
        all_words = self.read_data()
        c = collections.Counter(all_words)
        vocab_size = str(len(c) + 1) # the +1 is for UNK
        print("The size of the vocabulary is: {}.".format(vocab_size))


def main():
    argparser = configargparse.ArgumentParser(description="Create vector embedding for source code tokens.",
                                              default_config_files=["./config/config.default.yml"])

    argparser.add_argument("--mode", type=str, choices=["train", "vocab"],
                           help=("One of {train|vocab}, to "
                                 "indicate what you want the model to do. "
                                 "vocab: get the correct vocabulary size of the corpora."))
    argparser.add_argument("--name", type=str, default="default", help="Name for this run.")
    argparser.add_argument("--output_dir", type=str, default="output/")
    argparser.add_argument("--train_file", type=str, default="./data/token-vocabulary/small-tokens-vocab.txt")
    argparser.add_argument("--num_steps", type=int, default=100001, help="Number of training steps.")
    argparser.add_argument("--batch_size", type=int, default=128)
    argparser.add_argument("--embedding_size", type=int, default=8, help="Dimension of the embedding vector.")
    argparser.add_argument("--skip_window", type=int, default=1, help="How many words to consider left and right.")
    argparser.add_argument("--num_skips", type=int, default=2,
                           help="How many times to reuse an input to generate a label.")
    argparser.add_argument("--vocabulary_size", type=int, default=100, help="Size of the vocabulary.")
    argparser.add_argument("--valid_size", type=int, default=50, help="Random set of words to evaluate similarity on.")

    config = argparser.parse_args()

    embedding_size = config.embedding_size
    skip_window = config.skip_window
    num_skips = config.num_skips
    vocabulary_size = config.vocabulary_size

    path_of_config = "{}.dim-{}.vocab-{}.sw-{}.ns-{}".format(config.name, embedding_size, vocabulary_size, skip_window,
                                                             num_skips)
    output_path = os.path.join(config.output_dir, path_of_config)
    config.output_path = output_path

    config.token_index_file = os.path.join(output_path, "token_index.csv")
    config.count_of_words_file = os.path.join(output_path, "count.csv")
    config.token_embeddings_file = os.path.join(output_path, "embeddings.txt")
    ckpt_save_dir = os.path.join(output_path, "ckpt")
    config.ckpt_save_path = os.path.join(ckpt_save_dir, "ckpt")

    if not os.path.exists(ckpt_save_dir):
        os.makedirs(ckpt_save_dir)

    # data_setting = "small"
    # data_setting = "sample"
    #
    # if (data_setting == "small"):
    #     filename = './data/token-vocabulary/small-tokens-vocab.txt'
    #     # Step 2: Build the dictionary and replace rare words with UNK token.
    #     vocabulary_size = 70
    #     valid_size = 60  # Random set of words to evaluate similarity on.
    #     valid_window = vocabulary_size  # Only pick dev samples in the head of the distribution.
    # elif (data_setting == "sample"):
    #     filename = './data/token-vocabulary/sample-tokens-vocab-filtered.txt'
    #     # Step 2: Build the dictionary and replace rare words with UNK token.
    #     vocabulary_size = 107
    #     valid_size = 100  # Random set of words to evaluate similarity on.
    #     valid_window = vocabulary_size  # Only pick dev samples in the head of the distribution.

    w = Word2Vec(config)

    if config.mode == "train":
        w.train_data()
    else:
        w.test_vocab_size()


if __name__ == '__main__':
    main()
