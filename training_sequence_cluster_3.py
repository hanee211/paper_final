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

cluster_file = 'cluster_5000'

def learning():
	restore = True

	args = sys.argv
	args = args[1:]
	
	PAD = 0 
	EOS = 1
		
	for _i in range(int(len(args)/2)):
		arg_idx = _i * 2
		val_idx = _i * 2 + 1
		
		arg, value = args[arg_idx], args[val_idx]
		
		if arg == '-r':
			restore = value
			
	######################################			
	
	sentence_list, decoder_sentence_list, document_list  = wp.get_sentences_with_document_id()
	
	with open(cluster_file, 'rb') as file:
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
	
	stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
			labels=tf.one_hot(targets, depth=vocab_size, dtype=tf.float32),
			logits=logits,
	)

	loss = tf.reduce_mean(stepwise_cross_entropy)
	train_op = tf.train.AdamOptimizer().minimize(loss)	
	
	cluster_sentence = list()
	
	for c, d in zip(clusters.labels_, document_list[0:len(clusters.labels_)]):
		cluster_sentence.append((c + 2,d))

	m = max(cluster_sentence, key=lambda t:t[1])[1]
	

	cluster_seq_in_doc = [[] for i in range(m+1)]

	for c, d in cluster_sentence:
	    cluster_seq_in_doc[d].append(c)

	print("==== 3")	
	
	'''
	print(len(cluster_sentence))
	print(len(cluster_seq_in_doc))
	
	
	print(max([len(s) for s in cluster_seq_in_doc]))
	print([(i, len(s)) for i, s in enumerate(cluster_seq_in_doc) if len(s) > 50])
	print(cluster_seq_in_doc[0:10])
	
	'''
	saver = tf.train.Saver()
	
	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
		sess.run(tf.global_variables_initializer())
		model_ckpt_file = './status_cluster/model_cluster_seq.ckpt'
		
		if restore == 'T': 
			print("restoring.... ")
			saver.restore(sess, model_ckpt_file)		
		else:
			print("not restoring....")		
		
		
		max_batches = 500
		batches_in_epoch = 20
		batch_size = 128
		try:
			for e in range(max_batches):
				start_time = dt.datetime.now()
				for i in range(int(len(cluster_seq_in_doc)/batch_size) + 1):
					start = i * batch_size
					end = start + batch_size
					
					if end > len(cluster_seq_in_doc):
						end = len(cluster_seq_in_doc)				
					
					batch = cluster_seq_in_doc[start:end]
				
					input_, _ = helpers.batch(batch)
					target_, _ = helpers.batch(
						[(seq[1:]) + [EOS] for seq in batch]
					)					
					
					fd = {
						inputs: input_,
						targets : target_
					}						
					
					_, l = sess.run([train_op, loss], fd)
					
				print("Take", str((dt.datetime.now() - start_time).seconds), "seconds for ", str(e))

				if e == 0 or e % batches_in_epoch == 0:
					print('e {}'.format(e))
					print('  minibatch loss: {}'.format(sess.run(loss, fd)))
					predict_ = sess.run(prediction, fd)
					print(type(predict_))
					print(type(input_))
					
					for i, (inp, pred) in enumerate(zip(fd[inputs].T, predict_.T)):
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