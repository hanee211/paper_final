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

class GenerationModel():
	def __init__(self, params, training=True):
		print("Model Initialize")

		tf.reset_default_graph()
		#sess = tf.InteractiveSession()

		PAD = 0 
		EOS = 1

		vocab_size = params['vocab_size']
		input_embedding_size = params['input_embedding_size']
		hidden_units = params['encoder_hidden_units']
		batch_size = params['batch_size']
		

		self.inputs = tf.placeholder(shape=(None, None, 384), dtype=tf.float32, name='inputs')
		self.targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='targets')
		
		cell = tf.contrib.rnn.LSTMCell(hidden_units)

		_outputs, _ = tf.nn.dynamic_rnn(
			cell, self.inputs,
			dtype=tf.float32, time_major=True,
		)
		
		logits = tf.contrib.layers.linear(_outputs, vocab_size)
		self.prediction = tf.argmax(logits, 2)
		
		stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
				labels=tf.one_hot(self.targets, depth=vocab_size, dtype=tf.float32),
				logits=logits,
		)

		self.loss = tf.reduce_mean(stepwise_cross_entropy)
		self.train_op = tf.train.AdamOptimizer().minimize(self.loss)	

		
if __name__ == '__main__':
	params = dict()

	params['vocab_size'] = 10000
	params['input_embedding_size'] = 128 + 512 + 512
	params['encoder_hidden_units'] = 128
	params['batch_size'] = 32

	generation_model = GenerationModel(params)