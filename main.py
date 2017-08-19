from model import *
import parse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import time
sns.set()

data, vocab = parse.get_vocab('essay')
onehot = parse.embed_to_onehot(data, vocab)

# hyperparameters
learning_rate = 0.0001
length_sentence = 64
batch_size = 20
epoch = 100
num_layers = 2
size_layer = 512
len_noise = 100
possible_batch_id = range(len(data) - batch_size)

sess = tf.InteractiveSession()
model = Model(num_layers, size_layer, len(vocab), len_noise, length_sentence, learning_rate)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
sample_z = np.random.uniform(-1, 1, size = (length_sentence, len_noise)).reshape((1, length_sentence, len_noise))
start_tag = random.randint(0, len(data) - 3)
tag = data[start_tag: start_tag + 3]

DISC_LOSS, GEN_LOSS = [], []
for i in xrange(epoch):
	last_time = time.time()
	random_sample = np.random.uniform(-1, 1, size = (length_sentence, len_noise)).reshape((1, length_sentence, len_noise))
	batch_fake = np.zeros((length_sentence, batch_size, len(vocab)))
	batch_true = np.zeros((length_sentence, batch_size, len(vocab)))
	batch_id = random.sample(possible_batch_id, length_sentence)
	
	for n in xrange(batch_size):
		
		# fake sentence will randomly pick random word by word from the essay to learn how to become correct sentence
		id1 = [k + n for k in batch_id]
		batch_fake[:, n, :] = onehot[id1, :]
		
		# randomly pick any sub sentence from the essay
		start_random = random.randint(0, len(data) - length_sentence)
		batch_true[:, n, :] = onehot[start_random: start_random + length_sentence, :]
	
	disc_loss, _ = sess.run([model.d_loss, model.d_train_opt], feed_dict = {model.noise: random_sample, model.fake_input: batch_fake, model.true_sentence: batch_true})
	gen_loss, _ = sess.run([model.g_loss, model.g_train_opt], feed_dict = {model.noise: random_sample, model.fake_input: batch_fake, model.true_sentence: batch_true})
	
	# if the generator keep loosing too much, These steps may can help you:
	# 1- you may apply generator training for twice or more. 
	# 2- Tuned momentum constant in ADAM.
	# 3- change Adaptive RMS into pure RMS
	
	print 'epoch: ' + str(i + 1) + ', discriminator loss: ' + str(disc_loss) + ', generator loss: ' + str(gen_loss) + ', s/epoch: ' + str(time.time() - last_time)
	DISC_LOSS.append(disc_loss)
	GEN_LOSS.append(gen_loss)
	
	if (i + 1) % 5 == 0:
		print 'checkpoint: ' + str(i + 1)
		print 'generated sentence: '
		print generate(sess, length_sentence, sample_z, model, tag, length_sentence, vocab)
