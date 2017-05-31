import numpy as np
import tensorflow as tf
import helpers
import sys
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
from model import Model
import word_processing as wp
import myconf as cf
import pickle
from sklearn.cluster import KMeans


def Clustering():
	print("start getFinalState!!")

	PAD = 0 
	EOS = 1

	vocab_size = cf.vocab_size
	input_embedding_size = cf.input_embedding_size
	encoder_hidden_units = cf.encoder_hidden_units
	batch_size = cf.batch_size
	
	params = dict()
	params['vocab_size'] = vocab_size
	params['input_embedding_size'] = input_embedding_size
	params['encoder_hidden_units'] = encoder_hidden_units
	params['batch_size'] = batch_size
	
	model = Model(params, training = False)
	saver = tf.train.Saver()
	
	idx2word, word2idx = wp.get_wordListFromFile()
	sentences, decoder_target_sentence = wp.get_sentences()
	
	encoder_input_list = list()
	encoder_input_length_list = list()
	decoder_target_list = list()
	decoder_length_list = list()
		
	for i in range(int(len(sentences)/batch_size)):
		start = i * batch_size
		end = start + batch_size
		
		if end > len(sentences):
			end = len(sentences)

		batch = sentences[start:end]
		batch_decoder = decoder_target_sentence[start:end]
		
		encoder_inputs_, encoder_input_lengths_ = helpers.batch(batch)
		decoder_targets_, decoder_target_lengths_ = helpers.batch(batch_decoder)

		encoder_input_list.append(encoder_inputs_)
		encoder_input_length_list.append(encoder_input_lengths_)
		decoder_target_list.append(decoder_targets_)
		decoder_length_list.append(decoder_target_lengths_)

	
	final_state = list()	
	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
		sess.run(tf.global_variables_initializer())

		model_ckpt_file = './status/model.ckpt'
		saver.restore(sess, model_ckpt_file)
		print("Setting done.")		
		
		for i in range(int(len(sentences)/batch_size)):		
			fd = {
					model.encoder_inputs: encoder_input_list[i],
					model.encoder_inputs_length: encoder_input_length_list[i],
				}	
			
			_state = sess.run(model.encoder_final_state, fd)
				
			for j in range(len(_state)):
				#each_state = list()
				#each_state.extend(_state[0][j])
				#each_state.extend(_state[1][j])
				
				#final_state.append(each_state)
				
				final_state.append(_state[j][0:128])
			
		with open('final_state_vector', 'wb') as file:
			pickle.dump(final_state, file)
	
	
	print("start 500 Clustering!!")
	kmeans500 = KMeans(n_clusters=500, random_state=0).fit(final_state)
	with open('cluster_500', 'wb') as file:
		pickle.dump(kmeans500, file)	
	
	print("start 1000 Clustering!!")
	kmeans1000 = KMeans(n_clusters=1000, random_state=0).fit(final_state)
	with open('cluster_1000', 'wb') as file:
		pickle.dump(kmeans1000, file)	
			
if __name__ == '__main__':
	Clustering()		