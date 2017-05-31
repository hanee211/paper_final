import numpy as np
import tensorflow as tf
import helpers
import sys
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
from model import Model
import word_processing as wp
import myconf as cf
import pickle

saved_cluster_model = 'cluster_5000'

	
vocab_size = cf.vocab_size
input_embedding_size = cf.input_embedding_size
encoder_hidden_units = cf.encoder_hidden_units
batch_size = cf.batch_size

params = dict()
params['vocab_size'] = vocab_size
params['input_embedding_size'] = input_embedding_size
params['encoder_hidden_units'] = encoder_hidden_units
params['batch_size'] = batch_size	


args = sys.argv
args = args[1:]
	
for _i in range(int(len(args)/2)):
	arg_idx = _i * 2
	val_idx = _i * 2 + 1
	
	arg, value = args[arg_idx], args[val_idx]
	
	if arg == '-i':
		input_string = value
		
with tf.variable_scope("step1"):
	def get_vector(encoded_sentences):
		PAD = 0 
		EOS = 1

		model = Model(params, training = False)
		saver = tf.train.Saver()

		batch = encoded_sentences
		encoder_inputs_, encoder_input_lengths_ = helpers.batch(batch)
		
		with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
			sess.run(tf.global_variables_initializer())

			model_ckpt_file = './status/model.ckpt'
			saver.restore(sess, model_ckpt_file)
			

			fd = {
				model.encoder_inputs: encoder_inputs_ , 
				model.encoder_inputs_length: encoder_input_lengths_
			}	
			
			_state = sess.run(model.encoder_final_state, fd)
			return _state[0][0:128]
			

	def get_cluster(input_vector):
		with open('./' + saved_cluster_model, 'rb') as file:
			kmeans = pickle.load(file)	
		
		class_of_sentence = kmeans.predict([input_vector])[0] + 2
		return class_of_sentence

		
	'''
	--- initialize step ----
	get input text (seed text)

	## get vector from input text

	input_vector

	## get cluster from vector

	cluster_init
	--------------------------

	##
	prev_vector = init_vector
	input_vector = init_vector
	current_cluster = get_next_cluster(cluster_init)


	while
		current_cluster = get_next_cluster()
		
		input_ = current_cluster + prev_vector + input_vector

		##
		output_ = get_output_vector(input_)
		
		input_vector = output_

	'''
				
	print(input_string)
	input_string = wp.normalize_text(input_string)
	print(input_string)
	input_string = wp.sentence_encoding(input_string)
	print(input_string)
	
	#v = get_vector(input_string)
	#init_cluster = get_cluster(v)
	
	embedding = wp.get_wordEmbeddings()
	_, word2idx = wp.get_wordListFromFile()


	prev_vector = np.zeros(128)
	input_vector = embedding[word2idx['eos']]
	
	print(prev_vector)
	print(input_vector)
	
	
	
with tf.variable_scope("step2"):
	def get_next_cluster(start_class, num_class):
		
		with open('./' + saved_cluster_model, 'rb') as file:
			clusters = pickle.load(file)

		vocab_size = 1002
		input_embedding_size = 32

		hidden_units = 32

		inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='inputs')
		targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='targets')

		embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)
		inputs_embedded = tf.nn.embedding_lookup(embeddings, inputs)
		
		cell = tf.contrib.rnn.LSTMCell(hidden_units)

		_outputs, _ = tf.nn.dynamic_rnn(
			cell, inputs_embedded,
			dtype=tf.float32, time_major=True,
		)
		
		logits = tf.contrib.layers.linear(_outputs, vocab_size)
		prediction = tf.argmax(logits, 2)
		
		
		saver2 = tf.train.Saver()
		
		with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess2:
			sess2.run(tf.global_variables_initializer())
			
			model_ckpt_file = './status_cluster/model_cluster_seq.ckpt'
			saver2.restore(sess2, model_ckpt_file)		
			
			'''
			batch_ = list()
			l = [start_class]
			
			def next_feed():
				del batch_[:]
				batch_.append(l)
				
				print(batch_)
				encoder_inputs_, _ = helpers.batch(batch_)
				return  {
					inputs: encoder_inputs_,
				}			
				
			try:
				predict_ = ''
				for j in range(num_class):
					fd = next_feed()

					predict_ = sess.run(prediction, fd)
					for i, (inp, pred) in enumerate(zip(fd[inputs].T, predict_.T)):
						l.extend(predict_[-1])
						
						print('  sample {}:'.format(i + 1))
						print('    input     > {}'.format(inp))
						print('    predicted > {}'.format(pred))
						if i >= 6:
							break

				return predict_ .T
			except KeyboardInterrupt:
				print('training interrupted')	


		
		'''
	get_next_cluster(510, 5)