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

def predict():
	print("predict")
	
	vocab_size = cf.vocab_size
	input_embedding_size = cf.input_embedding_size
	hidden_units = cf.encoder_hidden_units * 3
	batch_size = cf.batch_size
	
	params = dict()
	params['vocab_size'] = vocab_size
	params['input_embedding_size'] = input_embedding_size
	params['encoder_hidden_units'] = hidden_units
	params['batch_size'] = batch_size
	
	model = GenerationModel(params)
	
	saver = tf.train.Saver()	

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		model_ckpt_file = './status_4th/model_4th.ckpt'
		saver.restore(sess, model_ckpt_file)		

		with open('variables', 'rb') as file:
			v = pickle.load(file)

		###  input_string_vector, init_cluster, zero_vector, eos_vector

		input_string_vector = v[0]
		zero_vector = v[2]
		eos_vector = v[3]

		input_item7 = list()
		
		with open('cluster_seq', 'rb') as file:
			c_seq = pickle.load(file)

		with open(cluster_file, 'rb') as file:
			clusters = pickle.load(file)	

		center_seq = [clusters.cluster_centers_[c] for c in c_seq]
	
		batch_ = list()
		#l = [start_class]
		#l = [eos_vector + center_seq[0] + input_string_vector][0]
		
		
		embeddings = wp.get_wordEmbeddings()
		idx2word, word2idx = wp.get_wordListFromFile()
	
		
		inner_item = list()
		#l = list(eos_vector) + list(center_seq[3]) + list(input_string_vector)
		l = list(embeddings[word2idx['eos']]) + list(center_seq[5]) + list(input_string_vector)
		inner_item.append(l)

		def next_feed():
			del batch_[:]
			print("inner item " , len(inner_item))
			batch_.append(inner_item)

			return  {
				model.inputs: np.array(batch_)
			}		
			
		try:
			predict_ = ""
			for j in range(20):
				fd = next_feed()
				
				predict_ = sess.run(model.prediction, fd)
				print(predict_ )

				input_item7.append(predict_[0][-1])
				
				l = list(embeddings[predict_[0][-1]]) + list(center_seq[5]) + list(input_string_vector)
				inner_item.append(l)
				print(len(inner_item))


			print(predict_)
			print(input_item7)
			print([idx2word[ii] for ii in predict_[0]])
		except KeyboardInterrupt:
			print('training interrupted')	

def learning():
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

	print(restore)	


	vocab_size = cf.vocab_size
	input_embedding_size = cf.input_embedding_size
	hidden_units = cf.encoder_hidden_units * 3
	batch_size = cf.batch_size

	
	params = dict()
	params['vocab_size'] = vocab_size
	params['input_embedding_size'] = input_embedding_size
	params['encoder_hidden_units'] = hidden_units
	params['batch_size'] = batch_size
	
	model = GenerationModel(params)
	
	saver = tf.train.Saver()	
	
	sentence_list, target_sentence_list, document_list  = wp.get_sentences_with_document_id_and_eos()
		
	with open(cluster_file, 'rb') as file:
		clusters = pickle.load(file)

	with open(state_vector_file, 'rb') as file:
		final_vector = pickle.load(file)		
		
	cluster_sentence = list()
	clusters_ = list()
	prev_vectors = list()
	
	prev = -1
	for i, (c,d) in enumerate(zip(clusters.labels_, document_list[0:len(clusters.labels_)])):
		if d != prev: # different, means, fill the vector with one
			#prev_vector = np.zeros(512)
			prev_vector = np.zeros(128)
		else:
			prev_vector = final_vector[i-1]
		
		#cluster_sentence.append((c,d, clusters.cluster_centers_[c], prev_vector))
		clusters_.append(clusters.cluster_centers_[c])
		prev_vectors.append(prev_vector)
		prev = d
	
	
	target_list = list()
	input_list = list()
	add_feature_list = list()
	
	max_len = len(final_vector)
	embeddings = wp.get_wordEmbeddings()

	
	for i in range(int(max_len/batch_size)):
		print("in data setting...",  i)
		
		start = i * batch_size
		end = start + batch_size
		
		if end > max_len:
			end = max_len
			
		if start == end:
			break
		
		targets_, _ = helpers.batch(target_sentence_list[start:end])
		inputs_, _ = helpers.batch(sentence_list[start:end])
		add_feature = [list(_x) + list(_y) for _x, _y in  zip(clusters_[start:end], prev_vectors[start:end])]
		
		target_list.append(targets_)
		input_list.append(inputs_)
		add_feature_list.append(add_feature)
		
	def next_feed(i):
		input_sentence_list = list()
		for _s in input_list[i]:
			input_word_list = list()
			for _i, _w in enumerate(_s):
				input_word_list.append(list(embeddings[_w]) + add_feature_list[i][_i])
		
			input_sentence_list.append(input_word_list)
		
		return input_sentence_list
		
	with tf.device('/cpu:0'):
		#with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			model_ckpt_file = './status_4th/model_4th.ckpt'
			
			print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
			if restore == 'T': 
				print("restoring.... ")
				saver.restore(sess, model_ckpt_file)		
			else:
				print("not restoring....")		
			
			
			max_batches = cf.max_batches
			batches_in_epoch = cf.batches_in_epoch
			
			try:
				for e in range(max_batches):
					start_time_out = dt.datetime.now()
					for i in range(int(max_len/batch_size)):
						start_time = dt.datetime.now()
						print("====> e = ", e , ",  i = ", i)
						print("1")

						fd = {
							model.inputs: np.array(next_feed(i)),
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