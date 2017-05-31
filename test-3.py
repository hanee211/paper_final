import numpy as np
import tensorflow as tf
import helpers
import sys
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
from model import Model
import word_processing as wp
import myconf as cf
import pickle
from model_generate_sentence import GenerationModel

cluster_file = 'cluster_1000'


vocab_size = cf.vocab_size
input_embedding_size = cf.input_embedding_size
encoder_hidden_units = cf.encoder_hidden_units
batch_size = cf.batch_size

params = dict()
params['vocab_size'] = vocab_size
params['input_embedding_size'] = input_embedding_size
params['encoder_hidden_units'] = encoder_hidden_units
params['batch_size'] = batch_size	

'''
with open('variables', 'rb') as file:
	v = pickle.load(file)

###  input_string_vector, init_cluster, zero_vector, eos_vector
	
input_string_vector = v[0]
zero_vector = v[2]
eos_vector = v[3]

with open('cluster_seq', 'rb') as file:
	c_seq = pickle.load(file)

with open(cluster_file, 'rb') as file:
	clusters = pickle.load(file)	
	
center_seq = [clusters.cluster_centers_[c] for c in c_seq]
	
'''

model = GenerationModel(params)	
saver = tf.train.Saver()
	
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
	sess.run(tf.global_variables_initializer())
	model_ckpt_file = './status_4th/model_4th.ckpt'

	saver.restore(sess, model_ckpt_file)		
	'''	
	batch_ = list()
	#l = [start_class]
	l = [eos_vector + center_seq[0] + input_string_vector]

	
	
	def next_feed():
		del batch_[:]
		batch_.append(l)
		
		print(batch_)
		encoder_inputs_, _ = helpers.batch(batch_)
		return  {
			model.inputs: encoder_inputs_,
		}			

		
	try:
		predict_ = ''
		for j in range(10):
			fd = next_feed()

			predict_ = sess.run(model.prediction, fd)
			for i, (inp, pred) in enumerate(zip(fd[inputs].T, predict_.T)):
				l.extend(eos_vector + center_seq[0] + predict_[-1])
				#l = [eos_vector + center_seq[0] + input_string_vector]
				
				print('  sample {}:'.format(i + 1))
				print('    input     > {}'.format(inp))
				print('    predicted > {}'.format(pred))
				if i >= 6:
					break

		#return predict_ .T
	except KeyboardInterrupt:
		print('training interrupted')
	'''