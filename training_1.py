import numpy as np
import tensorflow as tf
import helpers
import sys
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
from model import Model
import word_processing as wp
import myconf as cf
import datetime as dt

def train():
	print("start training!!")
	
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

	print(restore)	
	
	vocab_size = cf.vocab_size
	input_embedding_size = cf.input_embedding_size
	encoder_hidden_units = cf.encoder_hidden_units
	batch_size = cf.batch_size
	
	params = dict()
	params['vocab_size'] = vocab_size
	params['input_embedding_size'] = input_embedding_size
	params['encoder_hidden_units'] = encoder_hidden_units
	params['batch_size'] = batch_size
	
	model = Model(params)
	saver = tf.train.Saver()

	sentences, decoder_target_sentence = wp.get_sentences()

	encoder_input_list = list()
	encoder_input_length_list = list()
	decoder_target_list = list()
	decoder_length_list = list()
	

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

	
	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
	#with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		model_ckpt_file = './status/model.ckpt'

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
				print(e, " epoch start...")

				for i in range(int(len(sentences)/batch_size)):
					start_time = dt.datetime.now()

					fd = {
						model.encoder_inputs: encoder_input_list[i],
						model.encoder_inputs_length: encoder_input_length_list[i],
						model.decoder_targets: decoder_target_list[i],
						model.decoder_lengths : decoder_length_list[i]
					}						
					
					print("batch processing...")
					_, l = sess.run([model.train_op, model.loss], fd)
					print("Take", str((dt.datetime.now() - start_time).seconds), "seconds for ", str(i) , " in ", str(len(sentences)/batch_size))
				
				print("Take", str((dt.datetime.now() - start_time_out).seconds), "seconds for in epoch. current is ", str(e))
				if e == 0 or e % batches_in_epoch == 0:
					print('e {}'.format(e))
					print('  minibatch loss: {}'.format(sess.run(model.loss, fd)))
					predict_ = sess.run(model.decoder_prediction, fd)
					for i, (inp, pred) in enumerate(zip(fd[model.encoder_inputs].T, predict_.T)):
						print('  sample {}:'.format(i + 1))
						print('    input     > {}'.format(inp))
						print('    predicted > {}'.format(pred))
						if i >= 10:
							break
			
					saver.save(sess, model_ckpt_file)
					print("mode saved to ", model_ckpt_file)

		except KeyboardInterrupt:
			print('training interrupted')

if __name__ == '__main__':
	train()		