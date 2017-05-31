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
cluster_num = 0
first_sentence = True
args = sys.argv
args = args[1:]

PAD = 0 
EOS = 1
		
with open('loop_var', 'rb') as file:
	loop_var = pickle.load(file)
	
cluster_vector = loop_var['current_cluster_vector']
prev = loop_var['prev_sentence_vector']


vocab_size = cf.vocab_size
decoder_hidden_units = cf.decoder_hidden_units_4th
batch_size = cf.batch_size_4th
sentence_length = cf.sentence_length


params = dict()
params['vocab_size'] = vocab_size
vocab_size = int(vocab_size) + 1

params['decoder_hidden_units_4th'] = decoder_hidden_units
params['batch_size_4th'] = 2
params['sentence_length'] = sentence_length

embeddings = wp.get_wordEmbeddings_GO()
GO = vocab_size - 1


model = Model_4(params, False)	
saver = tf.train.Saver()	
	
with open(cluster_file, 'rb') as file:
	clusters = pickle.load(file)

with open(state_vector_file, 'rb') as file:
	final_vector = pickle.load(file)		
	
target_list = list()
target_length_list = list()
initial_state_c = list()
initial_state_h = list()
input_list = list()


max_len = len(final_vector)

#pickle.dump([input_string_vector, init_cluster, zero_vector, eos_vector], file)


print("----SSSSSSSSSSSSSSSSSSSSSSSSSSS--------")
print("-------------AAAAAAAAAAAAAAAAAAAAAAAA_------------")	

initial_state_c.append([list(cluster_vector), list(cluster_vector)])
initial_state_h.append([list(prev), list(prev)])
target_length_list.append([63,63])

tmp_list = list()
for j in range(2):
	in_tmp_list = list()
	for i in range(64):
		in_tmp_list.append(GO)
	tmp_list.append(in_tmp_list)
	
inputs_, _ = helpers.batch(tmp_list)
input_list.append(inputs_)

print("------BBBBBBBBBBBBBBBBBBBBBBBB---------")
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
				model.decoder_inputs : input_list[i],
				model.state_c : initial_state_c[i],
				model.state_h : initial_state_h[i],
			}						

			predict_ = sess.run(model.decoder_prediction, fd)

			
			pred = predict_.T
			pred = pred[0]
			print("-AAAAAAAAAAAAAAAAAAAAAAAA--")			
			print('    predicted > {}'.format(pred))
			
			pred_txt = [idx2word[idx] for idx in pred if idx != 0]
			generated_sentence = " ".join(pred_txt)
			print(generated_sentence)
			
			with open('generated_sentence', 'wb') as file:
				pickle.dump(generated_sentence, file)			

		except KeyboardInterrupt:
			print('training interrupted')

'''
if __name__ == '__main__':
	train()
'''