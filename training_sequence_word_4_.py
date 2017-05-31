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
	hidden_units = cf.encoder_hidden_units
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
	embedding_list = list()
	
	max_len = len(final_vector)
	embeddings = wp.get_wordEmbeddings()

	def next_feed(start, end)
	

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
		
		input_sentence_list = list()
		for _s in inputs_:
			input_word_list = list()
			for _i, _w in enumerate(_s):
				input_word_list.append(list(embeddings[_w]) + add_feature[_i])
		
			input_sentence_list.append(input_word_list)
		'''
		for s_ , af_ in zip(sentence_list[start:end], add_feature):
			t_ = [list(embeddings[word_id]) + af_ for word_id in s_]
			em_list.append(t_)
		em_list_ = np.transpose(em_list)
		'''
		embedding_list.append(input_sentence_list)
		target_list.append(targets_)
		
		print(len(input_sentence_list))
		print(len(targets_))
		
	print(len(embedding_list))
	print(len(target_list))
		
	print("data setting end")
	
	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
		sess.run(tf.global_variables_initializer())
		model_ckpt_file = './status_4th/model_4th.ckpt'
		
		max_batches = cf.max_batches
		batches_in_epoch = cf.batches_in_epoch
		
		try:
			for e in range(max_batches):
				start_time = dt.datetime.now()
				for i in range(int(max_len/batch_size)):
					
					fd = {
						model.inputs: np.array(embedding_list[i]),
						model.targets : target_list[i]
					}						
					
					_, l = sess.run([model.train_op, model.loss], fd)
					
				print("Take", str((dt.datetime.now() - start_time).seconds), "seconds for ", str(e))

				if e == 0 or e % batches_in_epoch == 0:
					print('e {}'.format(e))
					print('  minibatch loss: {}'.format(sess.run(model.loss, fd)))
					predict_ = sess.run(model.prediction, fd)
					print(type(predict_))
					
					for i, (inp, pred) in enumerate(zip(fd[model.inputs].T, predict_.T)):
						print('  sample {}:'.format(i + 1))
						print('    input     > {}'.format(inp))
						print('    predicted > {}'.format(pred))
						if i >= 20:
							break
					
					saver.save(sess, model_ckpt_file)
					print("mode saved to ", model_ckpt_file)
					
					
		except KeyboardInterrupt:
			print('training interrupted')	
	
if __name__ == '__main__':
	learning()