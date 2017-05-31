import numpy as np
import helpers
import sys
import tensorflow as tf
import word_processing as wp
import myconf as cf
from sklearn.cluster import KMeans
import helpers
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
import pickle
import datetime as dt
from model_generate_sentence import GenerationModel

cluster_file = 'cluster_1000'
state_vector_file = 'final_state_vector'

def train():
	PAD = 0
	EOS = 1
	
	restore = True

	args = sys.argv
	args = args[1:]
		
	for _i in range(int(len(args)/2)):
		arg_idx = _i * 2
		val_idx = _i * 2 + 1
		
		arg, value = args[arg_idx], args[val_idx]
		
		if arg == '-r':
			restore = value

	vocab_size = cf.vocab_size
	decoder_hidden_units = cf.decoder_hidden_units_4th
	batch_size = cf.batch_size_4th
	
	params = dict()
	params['vocab_size'] = vocab_size
	params['decoder_hidden_units_4th'] = decoder_hidden_units
	params['batch_size_4th'] = batch_size
	
	model = Model_4(params)
	
	saver = tf.train.Saver()	
	_, target_sentence_list, document_list  = wp.get_sentences_with_document_id_and_eos()
		
	with open(cluster_file, 'rb') as file:
		clusters = pickle.load(file)

	with open(state_vector_file, 'rb') as file:
		final_vector = pickle.load(file)		
		
	clusters_ = list()
	prev_vectors = list()
	
	prev = -1
	for i, (c,d) in enumerate(zip(clusters.labels_, document_list[0:len(clusters.labels_)])):
		if d != prev: # different, means, fill the vector with one
			prev_vector = np.zeros(128)
		else:
			prev_vector = final_vector[i-1]
		
		clusters_.append(clusters.cluster_centers_[c])
		prev_vectors.append(prev_vector)
		prev = d
	
	
	target_list = list()
	target_length_list = list()
	initial_state = list()
	
	max_len = len(final_vector)
	
	for i in range(int(max_len/batch_size)):
		start = i * batch_size
		end = start + batch_size
		
		if end > max_len:
			end = max_len
			
		if start == end:
			break
		
		targets_, target_length_ = helpers.batch(target_sentence_list[start:end])
		state_ = [list(_x) + list(_y) for _x, _y in  zip(clusters_[start:end], prev_vectors[start:end])]
		
		target_list.append(targets_)
		target_length_list.append(target_length_)
		initial_state.append(state_)
		
		
	def next_feed(i):
		input_sentence_list = list()
		for _s in input_list[i]:
			input_word_list = list()
			for _i, _w in enumerate(_s):
				input_word_list.append(list(embeddings[_w]) + add_feature_list[i][_i])
		
			input_sentence_list.append(input_word_list)
		
		return input_sentence_list
		
		
		
	for i in range(int(len(sentences)/batch_size) + 1):
		start = i * batch_size
		end = start + batch_size
		
		if end > len(sentences):
			end = len(sentences)
			
		if start == end:
			break

		batch = sentences[start:end]
		batch_decoder = decoder_target_sentence[start:end]
		
		encoder_inputs_, encoder_input_lengths_ = helpers.batch(batch)
		decoder_targets_, decoder_target_lengths_ = helpers.batch(batch_decoder)

		encoder_input_list.append(encoder_inputs_)
		encoder_input_length_list.append(encoder_input_lengths_)
		decoder_target_list.append(decoder_targets_)
		decoder_length_list.append(decoder_target_lengths_)



		
		
		

		
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		model_ckpt_file = './status_4th/model_4th.ckpt'
		
		if restore == 'T': 
			print(">>>> restoring.... ")
			saver.restore(sess, model_ckpt_file)		
		else:
			print(">>>> NOT restoring....")		
		
		max_batches = cf.max_batches
		batches_in_epoch = cf.batches_in_epoch
		
		try:
			for e in range(max_batches):
				start_time_out = dt.datetime.now()
				for i in range(int(max_len/batch_size)):
					start_time = dt.datetime.now()
					print("====> e = ", e , ",  i = ", i)

					fd = {
						model.targets : target_list[i]
					}						
					print("2")
					_, l = sess.run([model.train_op, model.loss], fd)
					print("Take", str((dt.datetime.now() - start_time).seconds), "seconds for ", i, "in e=", e)
					
				print("Take", str((dt.datetime.now() - start_time_out).seconds), "seconds for ", str(e))

				if e == 0 or e % batches_in_epoch == 0:
					print('e {}'.format(e))
					print('  minibatch loss: {}'.format(sess.run(model.loss, fd)))
					predict_ = sess.run(model.prediction, fd)
					print(type(predict_))
					
					for j, (inp, pred) in enumerate(zip(sentence_list[i * batch_size: (i + 1) * batch_size], predict_.T)):
						print('  sample {}:'.format(j + 1))
						print('    input     > {}'.format(inp))
						print('    predicted > {}'.format(pred))
						if j >= 30:
							break
					
					saver.save(sess, model_ckpt_file)
					print("mode saved to ", model_ckpt_file)
					
					
		except KeyboardInterrupt:
			print('training interrupted')	
		
if __name__ == '__main__':
	predict()