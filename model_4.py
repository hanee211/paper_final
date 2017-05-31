import numpy as np
import tensorflow as tf
import helpers
import sys
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, GRUCell, MultiRNNCell
import time
import word_processing as wp

class Model_4():
	def __init__(self, params, training=True):
		print("Model Initialize")

		tf.reset_default_graph()
		#sess = tf.InteractiveSession()

		PAD = 0 
		EOS = 1
		
		infer = False
		if training == False:
			infer = True
		
		print("training " , training)
		print("infer", infer)
		
		vocab_size = params['vocab_size']
		vocab_size = int(vocab_size) + 1
		
		decoder_hidden_units = params['decoder_hidden_units_4th']
		batch_size = params['batch_size_4th']
		sentence_length = int(params['sentence_length']) + 1
		embeddings = wp.get_wordEmbeddings_GO()
		print(len(embeddings))
		print(len(embeddings[0]))
		GO = vocab_size - 1

		self.decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')
		self.decoder_inputs = tf.placeholder(shape=(sentence_length, batch_size), dtype=tf.int32, name='decoder_inputs')
		
		#self.decoder_targets = tf.placeholder(tf.int32, [batch_size, seq_length])
		#self.decoder_inputs = tf.placeholder(tf.int32, [batch_size, seq_length])
		
		#-------------------------------------------------------
		#self.decoder_initial_state = tf.placeholder(shape=(None, decoder_hidden_units), dtype=tf.float32)
		#decoder_cell = GRUCell(decoder_hidden_units)
		#-------------------------------------------------------
		self.state_c = tf.placeholder(shape=(None, decoder_hidden_units), dtype=tf.float32)
		self.state_h = tf.placeholder(shape=(None, decoder_hidden_units), dtype=tf.float32)
		
		self.decoder_initial_state = LSTMStateTuple(
									c = self.state_c,
									h = self.state_h	)
		
		decoder_cell = LSTMCell(decoder_hidden_units)
		#decoder_cell = MultiRNNCell([decoder_cell] * 3, state_is_tuple=True)

		#-------------------------------------------------------
		
		self.decoder_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name='decoder_lengths')
		self.decoder_input_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='decoder_input_length')

		W = tf.Variable(tf.random_uniform([decoder_hidden_units, vocab_size], -1, 1), dtype=tf.float32)
		b = tf.Variable(tf.zeros([vocab_size]), dtype=tf.float32)

		#inputs = tf.nn.embedding_lookup(embeddings, self.decoder_inputs)
		#inputs = tf.nn.embedding_lookup(embeddings, input_format)
		#input_format = [self.decoder_inputs[ix] for ix in range(sentence_length)]

		inputs = [tf.nn.embedding_lookup(embeddings, self.decoder_inputs[ix]) for ix in range(sentence_length)]
		
		print(self.decoder_inputs[0])
		print(type(inputs))
		print(inputs)
		print(len(inputs))
		print(tf.shape(inputs[0]))
		
		#inputs = tf.split(1, sentence_length, tf.nn.embedding_lookup(embeddings, self.decoder_inputs))
		#inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
		
		def loop(prev, _):
			prev = tf.matmul(prev, W) + b
			prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
			return tf.nn.embedding_lookup(embeddings, prev_symbol)

		#outputs, last_state = tf.contrib.legacy_seq2seq.rnn_decoder(inputs, self.decoder_initial_state, decoder_cell, loop_function=loop if infer else None)		
		print("----2-----")
		decoder_outputs_ta, decoder_final_state = tf.contrib.legacy_seq2seq.rnn_decoder(inputs, self.decoder_initial_state, decoder_cell, loop_function=loop if infer else None)
		print("----3-----")
		#decoder_outputs = decoder_outputs_ta.stack()
		decoder_outputs = decoder_outputs_ta
		print("----4-----")
		
		#decoder_outputs = tf.reshape(decoder_outputs, [batch_size, sentence_length, decoder_hidden_units])
		x_for_softmax = tf.reshape(decoder_outputs, [-1, decoder_hidden_units])
		
		decoder_outputs_to_symbol = tf.matmul(x_for_softmax, W) + b
		decoder_outputs_to_symbol = tf.reshape(decoder_outputs_to_symbol, (sentence_length, batch_size, len(embeddings)))
		
		loss_weights = tf.ones([batch_size, sentence_length])
		
		self.loss = tf.contrib.seq2seq.sequence_loss(logits=decoder_outputs_to_symbol, targets=self.decoder_targets, weights=loss_weights)
		self.cost = tf.reduce_sum(self.loss) / batch_size / sentence_length
		
		self.train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.loss)
		self.decoder_prediction = tf.argmax(decoder_outputs_to_symbol, axis=2)		
		
		
		'''
		
		decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
		decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
		decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W), b)
		decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, vocab_size))

		self.decoder_prediction = tf.argmax(decoder_logits, 2)

		stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
			labels=tf.one_hot(self.decoder_targets, depth=vocab_size, dtype=tf.float32),
			logits=decoder_logits,
		)

		self.loss = tf.reduce_mean(stepwise_cross_entropy)
		self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
		'''
		
		
		
		
		
		'''
		output = tf.reshape(tf.concat(1, outputs), [-1, decoder_hidden_units])
		
		self.logits = tf.matmul(output, W) + b
		self.probs = tf.nn.softmax(self.logits)
		
		loss = seq2seq.sequence_loss_by_example([self.logits],
				[tf.reshape(self.decoder_targets, [-1])],
				[tf.ones([batch_size * seq_length])],
				vocab_size)
		self.cost = tf.reduce_sum(loss) / batch_size / seq_length		
		
		self.train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
		
		for_predict = tf.reshape(logits, [batch_size, seq_length, vocab_size])		
		self.decoder_prediction = tf.argmax(for_predict, axis=2)		
		'''
if __name__ == '__main__':
	params = dict()

	params['vocab_size'] = 10000
	params['decoder_hidden_units_4th'] = 128 * 2
	params['batch_size_4th'] = 128

	model = Model_4(params)