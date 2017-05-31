import numpy as np
import tensorflow as tf
import helpers
import sys
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
from model import Model
import word_processing as wp
import myconf as cf
import datetime as dt
from model_4 import Model_4
import pickle

cluster_file = 'cluster_5000'
state_vector_file = 'final_state_vector'


#def train():
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
decoder_hidden_units = cf.decoder_hidden_units_4th
batch_size = cf.batch_size_4th
sentence_length = cf.sentence_length

params = dict()
params['vocab_size'] = vocab_size
params['decoder_hidden_units_4th'] = decoder_hidden_units
params['batch_size_4th'] = batch_size
params['sentence_length'] = sentence_length




model = Model_4(params)	
saver = tf.train.Saver()	

decoder_input_sentence_list, target_sentence_list, document_list  = wp.get_sentences_with_document_id_and_eos_and_go()
	
with open(cluster_file, 'rb') as file:
	clusters = pickle.load(file)

with open(state_vector_file, 'rb') as file:
	final_vector = pickle.load(file)		
	
clusters_ = list()
prev_vectors = list()

prev = -1
for i, (c,d) in enumerate(zip(clusters.labels_, document_list[0:len(clusters.labels_)])):
	if d != prev: # different, means, fill the vector with one
		prev_vector = np.zeros(256)
	else:
		prev_vector = final_vector[i-1]
	
	clusters_.append(clusters.cluster_centers_[c])
	prev_vectors.append(prev_vector)
	prev = d

	

target_list = list()
target_length_list = list()

input_list = list()
input_length_list = list()

initial_state_c = list()
initial_state_h = list()

max_len = len(final_vector)

for i in range(int(max_len/batch_size)):
	start = i * batch_size
	end = start + batch_size
	
	if end > max_len:
		end = max_len
		
	if start == end:
		break
	
	targets_, target_length_ = helpers.batch(target_sentence_list[start:end])
	inputs_, input_length_ = helpers.batch(decoder_input_sentence_list[start:end])

	target_list.append(targets_)
	target_length_list.append(target_length_)
	
	input_list.append(inputs_)
	input_length_list.append(input_length_)
	
	initial_state_c.append(clusters_[start:end])
	initial_state_h.append(prev_vectors[start:end])

	#state_ = [list(c_) + list(p_) for c_, p_ in zip(clusters_[start:end], prev_vectors[start:end])]
	#initial_state.append(state_)	

with tf.device('/gpu:0'):
	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
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
				print(e, " epoch start...")

				start_time_out = dt.datetime.now()
				
				for i in range(int(max_len/batch_size)):
					start_time = dt.datetime.now()
					#print("====> epoch = ", e , ",  iteration = ", i)

					fd = {
						model.decoder_targets: target_list[i],
						model.decoder_lengths : target_length_list[i],
						#model.decoder_initial_state : initial_state[i], 
						
						model.decoder_inputs : input_list[i],
						model.decoder_input_length : input_length_list[i],
						
						model.state_c : initial_state_c[i],
						model.state_h : initial_state_h[i],
					}						
					#print("execute....")
					_, l = sess.run([model.train_op, model.loss], fd)
					#print("Take", str((dt.datetime.now() - start_time).seconds), "seconds for ", i, "in e=", e)
					
					
				print("Take", str((dt.datetime.now() - start_time_out).seconds), "seconds for in epoch. current is ", str(e))
				if e == 0 or e % batches_in_epoch == 0:
					print('e {}'.format(e))
					print('  minibatch loss: {}'.format(sess.run(model.loss, fd)))
					predict_ = sess.run(model.decoder_prediction, fd)
					print(type(predict_))
					target_string = np.array(target_list[i])
					print(target_string.T)
					for j, (inp, pred) in enumerate(zip(target_string.T, predict_.T)):
						print('  sample {}:'.format(j + 1))
						print('    input     > {}'.format(inp))
						print('    predicted > {}'.format(pred))
						if j >= 30:
							break
					if e % 20 == 0:
						saver.save(sess, model_ckpt_file)
						print("mode saved to ", model_ckpt_file)

		except KeyboardInterrupt:
			print('training interrupted')

'''
if __name__ == '__main__':
	train()
'''