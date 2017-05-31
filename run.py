import os
import sys
import numpy as np
import pickle

saved_cluster_model = 'cluster_5000'
result_file = 'result_sentence_cluster700_decoder_300'
result_dict = dict()

def main(num, input_sentence):

	result_dict[num] = list()
	

	'''
	args = sys.argv
	args = args[1:]
	
	input_sentence = ""
	
	for _i in range(int(len(args)/2)):
		arg_idx = _i * 2
		val_idx = _i * 2 + 1
		
		arg, value = args[arg_idx], args[val_idx]
		
		if arg == '-i':
			input_sentence = value
	'''
	
	with open('./' + saved_cluster_model, 'rb') as file:
		clusters = pickle.load(file)			
			
	print("input sentence : ", input_sentence)
	
	#1. get initial cluster from given sentence
	cmd1 = "python test-1-init_cluster.py -i \"" + input_sentence + "\""
	print(cmd1)
	os.system(cmd1)
	
	with open('variables', 'rb') as file:
		_va = pickle.load(file)
	
	prev_sentence_vector = _va['input_string_vector']

	
	#2. get cluster sequence from initial cluster 
	cmd2 = "python test-2-cluster-list.py -n 15"
	print(cmd2)
	os.system(cmd2)
	
	with open('cluster_seq', 'rb') as file:
		cluster_seq = pickle.load(file)
		
	print(cluster_seq)
	
	
	#3. loop with before sentence and cluster number
	for c_num in cluster_seq:
		current_cluster_vector = clusters.cluster_centers_[c_num]
		
		loop_var = dict()
		loop_var['current_cluster_vector'] = current_cluster_vector
		loop_var['prev_sentence_vector'] = prev_sentence_vector
		
		with open('loop_var', 'wb') as file:
			pickle.dump(loop_var, file)
		
		cmd3 = "python test-4-sentence-generation.py"
		os.system(cmd3)
		
		#################
		
		with open('generated_sentence', 'rb') as file:
			generated_sentence = pickle.load(file)
		
		result_dict[num].append(generated_sentence)
		
		cmd4 = "python test-1-init_cluster.py -i \"" + generated_sentence + "\""
		print(cmd4)
		os.system(cmd4)
		
		with open('variables', 'rb') as file:
			_va = pickle.load(file)
	
		prev_sentence_vector = _va['input_string_vector']
		
	with open(result_file, 'wb') as file:
		pickle.dump(result_dict, file)		
		

if __name__ == '__main__':
	main(0, "Record Intensity Hurricane Gilbert Causes Havoc In The Caribbean.")
	main(1, "Baffling, ferocious, Hurricane Gilbert roars through the Greater Antilles and strikes Mexico .")
	main(2, "Responses to the Northern California Earthquake of October 17, 1989 ")
	main(3, "Insurers will not suffer financial damage due to California earthquake.")
	main(4, "IRA blast kills 10 at Royal Marines School of Music.")
	main(5, "McDonald's opens restaurants in Korea, Yugoslavia, the Soviet Union, and China.")
	main(6, "A concerned world watches a divided Germany unite.")
	main(7, "Ferry capsizings from overloading, hitting reefs, high seas, etc., kill hundreds")
	main(8, "How the Hubble Space Telescope Finally Got Into Orbit")
	main(9, "John Lennon's spirit alive worldwide 10 years after his death.")
	main(10, "Death of Robert Maxwell raises questions of economic dealings.")
	main(11, "A verdict of innocent returned in the McMartin Pre-school trial")
	main(12 ,"Sam Walton dies. Master discounter founded and directed Wal-Mart.")
	main(13, "Insurers face large claims from Hurricane Andrew's assault on the south.")
	main(14, "New medical advances reduce the number of heart attacks nationwide.")
	main(15, "John Major elected leader of Conservative Party, becomes next Prime Minister.")
	main(16, "A survey of Britain's exciting Booker Prize for fiction novels.")