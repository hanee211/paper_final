import sys
import pickle
import os
import re

result_file = ""
output_file = ""

args = sys.argv
args = args[1:]


given_sentence = dict()

given_sentence[0] = "Record Intensity Hurricane Gilbert Causes Havoc In The Caribbean."
given_sentence[1] = "Baffling, ferocious, Hurricane Gilbert roars through the Greater Antilles and strikes Mexico ."
given_sentence[2] = "Responses to the Northern California Earthquake of October 17, 1989 "
given_sentence[3] = "Insurers will not suffer financial damage due to California earthquake."
given_sentence[4] = "IRA blast kills 10 at Royal Marines School of Music."
given_sentence[5] = "McDonald's opens restaurants in Korea, Yugoslavia, the Soviet Union, and China."
given_sentence[6] = "A concerned world watches a divided Germany unite."
given_sentence[7] = "Ferry capsizings from overloading, hitting reefs, high seas, etc., kill hundreds"
given_sentence[8] = "How the Hubble Space Telescope Finally Got Into Orbit"
given_sentence[9] = "John Lennon's spirit alive worldwide 10 years after his death."
given_sentence[10] = "Death of Robert Maxwell raises questions of economic dealings."
given_sentence[11] = "A verdict of innocent returned in the McMartin Pre-school trial"
given_sentence[12] = "Sam Walton dies. Master discounter founded and directed Wal-Mart."
given_sentence[13] = "Insurers face large claims from Hurricane Andrew's assault on the south."
given_sentence[14] = "New medical advances reduce the number of heart attacks nationwide."
given_sentence[15] = "John Major elected leader of Conservative Party, becomes next Prime Minister."
given_sentence[16] = "A survey of Britain's exciting Booker Prize for fiction novels."


for _i in range(int(len(args)/2)):
	arg_idx = _i * 2
	val_idx = _i * 2 + 1
	
	arg, value = args[arg_idx], args[val_idx]
	
	if arg == '-r':
		result_file = value
	elif arg == '-o':
		output_file = value


if os.path.exists(result_file) == False:
	print("File is not exsit")
	sys.exit(1)

with open(result_file, 'rb') as file:
	text = pickle.load(file)
	

def normalize_text(text):
	l = text
	l = l.strip()
	l = re.sub(r" eos", r"", l)
	l = re.sub(r"\. ", r"", l)
	l = re.sub(r"\\", r"", l)
	
	return l

f  = open(output_file, 'w')

for i in text:
	_text = text[i]
	
	f.write("------------------------------------------\n")
	f.write(given_sentence[i] + '\n')
	f.write("------------------------------------------\n")
	
	for txt in _text:
		t = normalize_text(txt)
		print(t)
		f.write(t)
		f.write("\n")
	
	f.write("\n")
		
	print("")
	print("===========================================")
	
f.close()
	
	
	
print("Program End")


