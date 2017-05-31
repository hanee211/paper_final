import numpy as np
import pickle
import myconf as cf
import html.parser
import html.parser as hp
from bs4 import BeautifulSoup
from pathlib2 import Path
import sys
import os
import re

word_file = './data/word'
embedding_file = './data/word_embedding'
sentence_file = './data/sentence_list'
	
sentence_length = cf.sentence_length

PAD = 0 
EOS = 1

def normalize_text(text):
	l = text
	l = l.strip()
	l = re.sub(r"\"", r" \" ", l)
	l = re.sub(r"``", r" \" ", l)
	l = re.sub(r"''", r" \" ", l)
	l = re.sub(r"\(", r"\( ", l)
	l = re.sub(r"\)", r" \)", l)
	l = re.sub(r", ", r" , ", l)	
	l = re.sub(r"'s", r" 's", l)	
	l = re.sub(r"!", r" !", l)
	l = re.sub(r"\?", r" \?", l)
	l = re.sub(r"\]", r" \]", l)
	l = re.sub(r"\[", r"\[ ", l)
	l = re.sub(r"\n\n\n\n", r"\n", l)
	l = re.sub(r"\n\n\n", r"\n", l)
	l = re.sub(r"\n\n", r"\n", l)
	l = re.sub(r":", r" :", l)
	l = re.sub(r";", r" ;", l)
	l = re.sub(r".<END>", r" . ", l)
	l = re.sub(r"    ", r" ", l)
	l = re.sub(r"   " , r" ", l)
	l = re.sub(r"  "  , r" ", l)
	l = l.lower()
	
	return l
	
def sentence_encoding(input_sentence):
	idx2word, word2idx = get_wordListFromFile()
	
	sentence_list = list()
	s_list = list()
	s_list.append(input_sentence)
		
	for _l in s_list:
		_l = _l.split(' ')
		
		t = list()
		decoder_t = list()
		
		for w in _l:
			w = w.strip()
			
			if w == '':
				continue
			
			if w in word2idx:
				t.append(word2idx[w])
			else:
				t.append(word2idx['UNK'])
		
		if len(t) <= sentence_length:
			padding_size = sentence_length - len(t)
			padding_vector = [PAD for i in range(padding_size)]
			
			t.extend(padding_vector)
			
			sentence_list.append(t)
			
	return sentence_list
	
	
	
def get_sentences():
	idx2word, word2idx = get_wordListFromFile()
	
	sentence_list = list()
	decoder_sentence_list = list()
	
	with open(sentence_file, 'rb') as file:
		s_list = pickle.load(file)
		
	for _l in s_list:
		_l = _l[1]
		_l = _l.split(' ')
		
		t = list()
		decoder_t = list()
		
		for w in _l:
			w = w.strip()
			
			if w == '':
				continue
			
			if w in word2idx:
				t.append(word2idx[w])
				decoder_t.append(word2idx[w])
			else:
				t.append(word2idx['UNK'])
				decoder_t.append(word2idx['UNK'])
		
		decoder_t.append(EOS)
		
		if len(t) <= sentence_length:
			padding_size = sentence_length - len(t)
			padding_vector = [PAD for i in range(padding_size)]
			
			t.extend(padding_vector)
			decoder_t.extend(padding_vector)
			
			sentence_list.append(t)
			decoder_sentence_list.append(decoder_t)
			
	return sentence_list, decoder_sentence_list
	
def get_sentences_with_document_id():
	idx2word, word2idx = get_wordListFromFile()
	
	document_list = list()
	sentence_list = list()
	decoder_sentence_list = list()
	
	with open(sentence_file, 'rb') as file:
		s_list = pickle.load(file)
		
	for _l in s_list:
		dc = _l[0]
		_l = _l[1]
		_l = _l.split(' ')
		
		t = list()
		decoder_t = list()
		
		for w in _l:
			w = w.strip()
			
			if w == '':
				continue
			
			if w in word2idx:
				t.append(word2idx[w])
				decoder_t.append(word2idx[w])
			else:
				t.append(word2idx['UNK'])
				decoder_t.append(word2idx['UNK'])
		
		decoder_t.append(EOS)
		
		if len(t) <= sentence_length:
			padding_size = sentence_length - len(t)
			padding_vector = [PAD for i in range(padding_size)]
			
			t.extend(padding_vector)
			decoder_t.extend(padding_vector)
			
			sentence_list.append(t)
			decoder_sentence_list.append(decoder_t)
			document_list.append(dc)
			
	return sentence_list, decoder_sentence_list	, document_list

def get_sentences_with_document_id_and_eos_and_go():
	idx2word, word2idx = get_wordListFromFile()
	GO = len(idx2word)
	
	document_list = list()
	sentence_list = list()
	decoder_sentence_list = list()
	
	with open(sentence_file, 'rb') as file:
		s_list = pickle.load(file)
		
	for _l in s_list:
		dc = _l[0]
		_l = _l[1]
		_l = _l.split(' ')
		
		t = list()
		decoder_t = list()
		
		for w in _l:
			w = w.strip()
			
			if w == '':
				continue
			
			if w in word2idx:
				t.append(word2idx[w])
				decoder_t.append(word2idx[w])
			else:
				t.append(word2idx['UNK'])
				decoder_t.append(word2idx['UNK'])
		
		decoder_t.append(EOS)
		
		if len(t) <= sentence_length:
			padding_size = sentence_length - len(t)
			padding_vector = [PAD for i in range(padding_size)]
			
			t.extend(padding_vector)
			t = [GO] + t
			decoder_t.extend(padding_vector)
			
			
			sentence_list.append(t)
			decoder_sentence_list.append(decoder_t)
			document_list.append(dc)
			
	return sentence_list, decoder_sentence_list	, document_list
	
	
def get_sentences_with_document_id_and_eos():
	idx2word, word2idx = get_wordListFromFile()
	
	document_list = list()
	sentence_list = list()
	decoder_sentence_list = list()
	
	with open(sentence_file, 'rb') as file:
		s_list = pickle.load(file)
		
	for _l in s_list:
		dc = _l[0]
		_l = _l[1]
		_l = _l.split(' ')
		
		t = list()
		decoder_t = list()
		
		for w in _l:
			w = w.strip()
			
			if w == '':
				continue
			
			if w in word2idx:
				t.append(word2idx[w])
				decoder_t.append(word2idx[w])
			else:
				t.append(word2idx['UNK'])
				decoder_t.append(word2idx['UNK'])
		
		decoder_t.append(EOS)
		
		if len(t) <= sentence_length:
			padding_size = sentence_length - len(t)
			padding_vector = [PAD for i in range(padding_size)]
			
			t.extend(padding_vector)
			t = [EOS] + t
			decoder_t.extend(padding_vector)
			
			
			sentence_list.append(t)
			decoder_sentence_list.append(decoder_t)
			document_list.append(dc)
			
	return sentence_list, decoder_sentence_list	, document_list
	
def get_wordEmbeddings():
	with open(embedding_file, 'rb') as file:
		embedding = pickle.load(file)
		
	return embedding
	
def get_wordEmbeddings_GO():
	with open(embedding_file, 'rb') as file:
		embedding_go = pickle.load(file)
		
	c = [[ 0.43269549,  0.91172722,  0.86304956,  0.13523882,  0.96300839,
		0.07983022,  0.46861899,  0.31473584,  0.70324425,  0.42322007,
        0.68175376,  0.17948741,  0.39063382,  0.84804543,  0.00434547,
        0.05242531,  0.89513646,  0.14345666,  0.25686128,  0.25491479,
        0.80464097,  0.66463692,  0.83945773,  0.66894903,  0.68210802,
        0.97145398,  0.24544364,  0.97637949,  0.14186873,  0.12597699,
        0.80449171,  0.50923087,  0.10297539,  0.33057242,  0.1000756 ,
        0.76301135,  0.25602352,  0.33527646,  0.72191115,  0.56012385,
        0.38529957,  0.15774249,  0.05196414,  0.497801  ,  0.41420946,
        0.09169882,  0.66235825,  0.58070357,  0.39107187,  0.86210803,
        0.41445568,  0.65726777,  0.27057969,  0.61744253,  0.16306876,
        0.41130656,  0.20217943,  0.51336825,  0.49394946,  0.71678717,
        0.49198377,  0.77205155,  0.40820828,  0.30652549,  0.35180691,
        0.84713729,  0.97745788,  0.58376174,  0.13529329,  0.0748293 ,
		0.91757061,  0.90703667,  0.38815218,  0.57999927,  0.44495762,
        0.32411716,  0.39826464,  0.43476584,  0.58073795,  0.33306956,
        0.69450073,  0.77337057,  0.66385598,  0.60718764,  0.22625134,
        0.29748775,  0.45666738,  0.03077786,  0.20781893,  0.96671772,
        0.14389396,  0.46487861,  0.88869961,  0.2175513 ,  0.81537093,
        0.85168377,  0.88259446,  0.25459518,  0.64621109,  0.93982709,
        0.84547683,  0.33934593,  0.4971436 ,  0.05093994,  0.08269146,
        0.77773466,  0.67380857,  0.35544348,  0.21479651,  0.03094952,
        0.5387494 ,  0.90207855,  0.38009663,  0.79052117,  0.67414772,
        0.06659741,  0.06658323,  0.55488797,  0.63642454,  0.13008525,
        0.32567073,  0.90365037,  0.45998755,  0.72785679,  0.243241  ,
        0.03903368,  0.62067196,  0.09464498]]

	go_em = np.array(c)
	embedding_go = np.append(embedding_go, go_em, axis=0)
	
	embedding_go = embedding_go.astype(np.float32)
	return embedding_go

def get_wordListFromFile():
	with open(word_file, 'rb') as file:
		idx2word = pickle.load(file)
			
	word2idx = {w:i for i,w in enumerate(idx2word)}
		
	return idx2word, word2idx

if __name__ == '__main__':
	'''
	sentences = get_sentences()
	print(sentences[0])
	print(sentences[1])
	print(sentences[2])
	print(len(sentences))
	'''
