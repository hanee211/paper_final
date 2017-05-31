import numpy as np
import tensorflow as tf
import helpers
import sys
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, GRUCell
import time
import word_processing as wp

class Model():
	def __init__(self, params, training=True):
		print("Model Initialize")

		tf.reset_default_graph()
		#sess = tf.InteractiveSession()

		PAD = 0 
		EOS = 1

		vocab_size = params['vocab_size']
		input_embedding_size = params['input_embedding_size']

		encoder_hidden_units = params['encoder_hidden_units']
		decoder_hidden_units = encoder_hidden_units * 2
		
		batch_size = params['batch_size']
		
		#if training:
		self.encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
		self.encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')
		self.decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')
		
		#embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)
		embeddings = wp.get_wordEmbeddings()
		encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, self.encoder_inputs)

		encoder_cell = GRUCell(encoder_hidden_units)
		decoder_cell = GRUCell(decoder_hidden_units)
		
		((encoder_fw_outputs, encoder_bw_outputs), 
			(encoder_fw_final_state, encoder_bw_final_state)) = (
				tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell, cell_bw = encoder_cell, inputs=encoder_inputs_embedded, sequence_length=self.encoder_inputs_length, dtype=tf.float32, time_major=True)
			)
			
		encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs),2)

		##encoder_final_state_c = tf.concat((encoder_fw_final_state.c, encoder_bw_final_state.c), 1)
		##encoder_final_state_h = tf.concat((encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

		self.encoder_final_state = tf.concat((encoder_fw_final_state, encoder_bw_final_state), 1)

		
		##self.encoder_final_state = LSTMStateTuple(
		##						c = encoder_final_state_c,
		##						h = encoder_final_state_h)
		
		
		#else:
		##self.state_c = tf.placeholder(shape=(None, encoder_hidden_units * 2), dtype=tf.float32)
		##self.state_h = tf.placeholder(shape=(None, encoder_hidden_units * 2), dtype=tf.float32)
		
		self.direct_encoder_final_state = tf.placeholder(shape=(None, encoder_hidden_units * 2), dtype=tf.float32)
		
		
		##self.direct_encoder_final_state = LSTMStateTuple(
		##							c = self.state_c,
		##							h = self.state_h	)
		
		#encoder_max_time, batch_size = tf.unstack(tf.shape(self.encoder_inputs))

		self.decoder_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name='decoder_lengths')


		W = tf.Variable(tf.random_uniform([decoder_hidden_units, vocab_size], -1, 1), dtype=tf.float32)
		b = tf.Variable(tf.zeros([vocab_size]), dtype=tf.float32)


		assert EOS == 1 and PAD == 0

		eos_time_slice = tf.ones([batch_size], dtype=tf.int32, name='EOS')
		pad_time_slice = tf.zeros([batch_size], dtype=tf.int32, name='PAD')


		eos_step_embedded = tf.nn.embedding_lookup(embeddings, eos_time_slice)
		pad_step_embedded = tf.nn.embedding_lookup(embeddings, pad_time_slice)

		print("##############################################################")
		print("##############################################################")
		
		if training:
			print ("Traing...")
		else:
			print("Test....")

		print("##############################################################")
		print("##############################################################")

		def loop_fn_initial():
			initial_elements_finished = (0 >= self.decoder_lengths)
			initial_input = eos_step_embedded  
			#initial_cell_state = encoder_final_state
			initial_cell_state = self.encoder_final_state
			initial_cell_output = None
			initial_loop_state = None
			
			return (initial_elements_finished,
					initial_input,
					initial_cell_state,
					initial_cell_output,
					initial_loop_state)

		def loop_fn_initial_direct():
			initial_elements_finished = (0 >= self.decoder_lengths)
			initial_input = eos_step_embedded  
			#initial_cell_state = encoder_final_state
			initial_cell_state = self.direct_encoder_final_state
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
				if training:
					print("in loop_fn, training....")
					return loop_fn_initial()
				else:
					print("in loop_fn, testing....")
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
	params['input_embedding_size'] = 128 + 512 + 512
	params['encoder_hidden_units'] = 128
	params['batch_size'] = 32
	
	model = Model(params)