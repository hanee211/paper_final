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

##############################################################################################################################################
#
#   python test-2-cluster-list.py -n 15
#
##############################################################################################################################################

	
vocab_size = cf.vocab_size
input_embedding_size = cf.input_embedding_size
encoder_hidden_units = cf.encoder_hidden_units
batch_size = cf.batch_size

params = dict()
params['vocab_size'] = vocab_size
params['input_embedding_size'] = input_embedding_size
params['encoder_hidden_units'] = encoder_hidden_units
params['batch_size'] = batch_size	

def get_next_cluster(start_class, num_class):
	
	with open('./' + saved_cluster_model, 'rb') as file:
		clusters = pickle.load(file)

	vocab_size = 5002
	input_embedding_size = 128

	hidden_units = 128

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
	
	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
		sess.run(tf.global_variables_initializer())

				
		model_ckpt_file = './status_cluster/model_cluster_seq.ckpt'
		saver2.restore(sess, model_ckpt_file)		
		

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
					#if i >= 6:
					#	break

			return predict_ .T
		except KeyboardInterrupt:
			print('training interrupted')	


if __name__ == '__main__':
	args = sys.argv
	args = args[1:]
		
	for _i in range(int(len(args)/2)):
		arg_idx = _i * 2
		val_idx = _i * 2 + 1
		
		arg, value = args[arg_idx], args[val_idx]
		
		if arg == '-s':
			start_class = int(value)
		elif arg == '-n':
			num_class = int(value)
			
	with open('variables', 'rb') as file:
		_va = pickle.load(file)
	
			
	###  input_string_vector, init_cluster, zero_vector, eos_vector

	class_seq = get_next_cluster(_va['init_cluster'], num_class)
	print("#####################################")
	print(class_seq[0])
	
	with open('cluster_seq', 'wb') as file:
		pickle.dump(class_seq[0], file)
