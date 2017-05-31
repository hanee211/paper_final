import numpy as np
import tensorflow as tf
import helpers
import sys
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, GRUCell
import time
import word_processing as wp

class Model_4():
	def __init__(self, params, training=True):
		print("Model Initialize")

		tf.reset_default_graph()
		#sess = tf.InteractiveSession()

		PAD = 0 
		EOS = 1

		vocab_size = params['vocab_size']
		decoder_hidden_units = params['decoder_hidden_units_4th']
		batch_size = params['batch_size_4th']	
		

		self.decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')
		self.decoder_initial_state = tf.placeholder(shape=(None, decoder_hidden_units), dtype=tf.float32)
		
		
		decoder_cell = GRUCell(decoder_hidden_units)
		self.decoder_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name='decoder_lengths')

		W = tf.Variable(tf.random_uniform([decoder_hidden_units, vocab_size], -1, 1), dtype=tf.float32)
		b = tf.Variable(tf.zeros([vocab_size]), dtype=tf.float32)


		assert EOS == 1 and PAD == 0

		eos_time_slice = tf.ones([batch_size], dtype=tf.int32, name='EOS')
		pad_time_slice = tf.zeros([batch_size], dtype=tf.int32, name='PAD')

		embeddings = wp.get_wordEmbeddings()
		eos_step_embedded = tf.nn.embedding_lookup(embeddings, eos_time_slice)
		pad_step_embedded = tf.nn.embedding_lookup(embeddings, pad_time_slice)

		def loop_fn_initial_direct():
			initial_elements_finished = (0 >= self.decoder_lengths)
			initial_input = eos_step_embedded  
			#initial_cell_state = encoder_final_state
			initial_cell_state = self.decoder_initial_state
			initial_cell_output = None
			initial_loop_state = None
			
			return (initial_elements_finished,
					initial_input,
					initial_cell_state,
					initial_cell_output,
					initial_loop_state)					

					
		def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):	
			def get_next_input():
				output_logits = tf.add(tf.matmul(previous_output, W), b)
				prediction = tf.argmax(output_logits, axis=1)
				next_input = tf.nn.embedding_lookup(embeddings, prediction)
				return next_input
				
			elements_finished = (time >= self.decoder_lengths) 
			finished = tf.reduce_all(elements_finished)
			#input = tf.cond(finished, lambda : pad_step_embedded, get_next_input)
			input = get_next_input()
			state = previous_state
			output = previous_output
			loop_state = None
			
			return(elements_finished, input, state, output, loop_state)
			
		def loop_fn(time, previous_output, previous_state, previous_loop_state):
			if previous_state is None:
				assert previous_output is None and previous_state is None
				return loop_fn_initial_direct()
			else:
				return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)

		decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
		decoder_outputs = decoder_outputs_ta.stack()


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
		#train_op_rms = tf.train.RMSPropOptimizer(learning_rate = 0.01).minimize(loss)
		#train_op_gd = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(loss)
		#train_op_ad = tf.train.AdagradOptimizer(learning_rate = 0.01).minimize(loss)
		#train_op_mt= tf.train.MomentumOptimizer(learning_rate = 0.01, ).minimize(loss)
		
if __name__ == '__main__':
	params = dict()

	params['vocab_size'] = 10000
	params['decoder_hidden_units_4th'] = 128 * 2
	params['batch_size_4th'] = 128

	model = Model_4(params)