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
#   python test-1-init_cluster.py -i "Baffling, ferocious, Hurricane Gilbert roars through the Greater Antilles and strikes Mexico ."
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
		#return _state[0][0:128]
		return _state[0]
		

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

if __name__ == '__main__':
	args = sys.argv
	args = args[1:]
		
	input_string = ""
	for _i in range(int(len(args)/2)):
		arg_idx = _i * 2
		val_idx = _i * 2 + 1
		
		arg, value = args[arg_idx], args[val_idx]
		
		if arg == '-i':
			input_string = value
			
			
	print(input_string)
	
	input_string = wp.normalize_text(input_string)
	print(input_string)
	
	input_string = wp.sentence_encoding(input_string)
	print(input_string)
	
	input_string_vector = get_vector(input_string)
	init_cluster = get_cluster(input_string_vector)
	
	#print(input_string_vector)
	#print(len(input_string_vector))
	#print(init_cluster)
	
	embedding = wp.get_wordEmbeddings()
	_, word2idx = wp.get_wordListFromFile()

	zero_vector = np.zeros(128)
	eos_vector = embedding[word2idx['eos']]
	
	#print(zero_vector)
	#print(eos_vector)
	
	_va = dict()
	_va['init_cluster'] = init_cluster
	_va['input_string_vector'] = input_string_vector
	_va['eos_vector'] = eos_vector
	
	with open('variables', 'wb') as file:
		#pickle.dump([input_string_vector, init_cluster, zero_vector, eos_vector], file)
		pickle.dump(_va, file)
	
	