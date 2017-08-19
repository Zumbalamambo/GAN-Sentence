import tensorflow as tf
import numpy as np
import parse

def lstm_cell(size_layer, state_is_tuple = True):
	return tf.nn.rnn_cell.LSTMCell(size_layer, state_is_tuple = state_is_tuple)

def generate(sess, sequence, noise, model, tag, length_sentence, text_vocab):
	sentence_generated = tag
	onehot = parse.embed_to_onehot(tag, text_vocab)
	for i in xrange(len(tag)):
		batch_fake = np.zeros((sequence, 1, len(text_vocab)))
		batch_fake[:, 0, :] = onehot[i, :]
		prob = sess.run(model.final_outputs, feed_dict = {model.noise: noise, model.fake_input: batch_fake})
	
	for i in xrange(length_sentence):
		word = np.random.choice(range(len(text_vocab)), p = prob[0][0])
		element = text_vocab[word]
		sentence_generated.append(element)
		onehot = parse.embed_to_onehot([element], text_vocab)
		batch_fake = np.zeros((sequence, 1, len(text_vocab)))
		batch_fake[:, 0, :] = onehot[0, :]
		prob = sess.run(model.final_outputs, feed_dict = {model.noise: noise, model.fake_input: batch_fake})
    
	return ' '.join(sentence_generated)


def discriminator(X, num_layers, size_layer, dimension_input, reuse = False):
	with tf.variable_scope("discriminator", reuse = reuse):
		rnn_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(size_layer) for _ in xrange(num_layers)])
		outputs, last_state = tf.nn.dynamic_rnn(rnn_cells, X, dtype = tf.float32)
		rnn_W = tf.Variable(tf.random_normal((size_layer, 1)))
		rnn_B = tf.Variable(tf.random_normal([1]))
		return tf.matmul(outputs[-1], rnn_W) + rnn_B
	
def generator_encode(X, num_layers, size_layer, len_noise, reuse = False):
	with tf.variable_scope("generator_encode", reuse = reuse):
		rnn_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(size_layer, False) for _ in xrange(num_layers)], state_is_tuple = False)
		_, final_state = tf.nn.dynamic_rnn(rnn_cells, X, dtype = tf.float32, time_major = True)
		return final_state
	
def generator_sentence(X, hidden_layer, num_layers, size_layer, dimension_input):
	with tf.variable_scope("generator_sentence", reuse = False):
		rnn_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(size_layer, False) for _ in xrange(num_layers)], state_is_tuple = False)
		outputs, last_state = tf.nn.dynamic_rnn(rnn_cells, X, initial_state = hidden_layer, dtype = tf.float32)
		seq_shape = tf.shape(outputs)
		rnn_W = tf.Variable(tf.random_normal((size_layer, dimension_input)))
		rnn_B = tf.Variable(tf.random_normal([dimension_input]))
		logits = tf.matmul(tf.reshape(outputs, [-1, size_layer]), rnn_W) + rnn_B
		return tf.reshape(tf.nn.softmax(logits), (seq_shape[0], seq_shape[1], dimension_input))
	
class Model:
	def __init__(self, num_layers, size_layer, dimension_input, len_noise, sequence_size, learning_rate):
		self.noise = tf.placeholder(tf.float32, [None, None, len_noise])
		self.fake_input = tf.placeholder(tf.float32, [None, None, dimension_input])
		self.true_sentence = tf.placeholder(tf.float32, [None, None, dimension_input])
		self.initial_layer = generator_encode(self.noise, num_layers, size_layer, len_noise)
		self.final_outputs = generator_sentence(self.fake_input, self.initial_layer, num_layers, size_layer, dimension_input)
		fake_logits = discriminator(self.final_outputs, num_layers, size_layer, dimension_input)
		true_logits = discriminator(self.true_sentence, num_layers, size_layer, dimension_input, reuse = True)
		d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = true_logits, labels = tf.ones_like(true_logits)))
		d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = fake_logits, labels = tf.zeros_like(fake_logits)))
		self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = fake_logits, labels = tf.ones_like(fake_logits)))
		
		self.d_loss = d_loss_real + d_loss_fake
		d_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'discriminator')
		g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'generator_encode') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'generator_sentence')
		self.d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1 = 0.5).minimize(self.d_loss, var_list = d_vars)
		self.g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1 = 0.5).minimize(self.g_loss, var_list = g_vars)