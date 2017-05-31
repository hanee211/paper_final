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

cluster_file = 'cluster_1000'
state_vector_file = 'final_state_vector'


#def train():
print("start training!!")

restore = True
cluster_num = 0
first_sentence = True
args = sys.argv
args = args[1:]

PAD = 0 
EOS = 1
	
for _i in range(int(len(args)/2)):
	arg_idx = _i * 2
	val_idx = _i * 2 + 1
	
	arg, value = args[arg_idx], args[val_idx]
	
	if arg == '-c':
		cluster_num = int(value)
	elif arg == '-f':
		if value == 'T':
			first_sentence = True
		else:
			first_sentence = False
		

print(restore)	

vocab_size = cf.vocab_size
decoder_hidden_units = cf.decoder_hidden_units_4th
batch_size = cf.batch_size_4th


params = dict()
params['vocab_size'] = vocab_size
params['decoder_hidden_units_4th'] = decoder_hidden_units
params['batch_size_4th'] = 2

model = Model_4(params)	
saver = tf.train.Saver()	
	
with open(cluster_file, 'rb') as file:
	clusters = pickle.load(file)

with open(state_vector_file, 'rb') as file:
	final_vector = pickle.load(file)		
	
target_list = list()
target_length_list = list()
initial_state_c = list()
initial_state_h = list()

max_len = len(final_vector)

#pickle.dump([input_string_vector, init_cluster, zero_vector, eos_vector], file)


with open('variables', 'rb') as file:
	clus = pickle.load(file)
	
if first_sentence == True:
	prev = np.zeros([256])
else:
	prev = clus[0]
	
cluster_vector = clusters.cluster_centers_[cluster_num]


print("----SSSSSSSSSSSSSSSSSSSSSSSSSSS--------")
print("-------------AAAAAAAAAAAAAAAAAAAAAAAA_------------")	

initial_state_c.append([list(cluster_vector), list(cluster_vector)])
initial_state_h.append([list(prev), list(prev)])
target_length_list.append([63,63])


print("------BBBBBBBBBBBBBBBBBBBBBBBB---------")
print(len(clus[0]))
print(len(prev))

print("------DDDDDDDDDDDDDDDDDDDDDDDD---------")

idx2word,  word2idx = wp.get_wordListFromFile()

with tf.device('/gpu:0'):
	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
		sess.run(tf.global_variables_initializer())
		model_ckpt_file = './status_4th/model_4th.ckpt'
		
		print(">>>> restoring.... ")
		saver.restore(sess, model_ckpt_file)		

		
		i = 0
		try:
			fd = {
				model.decoder_lengths : target_length_list[i],
				model.state_c : initial_state_c[i],
				model.state_h : initial_state_h[i],
			}						

			predict_ = sess.run(model.decoder_prediction, fd)

			
			pred = predict_.T
			pred = pred[0]
			print("-AAAAAAAAAAAAAAAAAAAAAAAA--")			
			print('    predicted > {}'.format(pred))
			
			pred_txt = [idx2word[idx] for idx in pred if idx != 0]
			print(" ".join(pred_txt))

		except KeyboardInterrupt:
			print('training interrupted')

'''
if __name__ == '__main__':
	train()
'''