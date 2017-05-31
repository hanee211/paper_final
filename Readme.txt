-----------------------------------------------------
INFORMATION

setting file : myconf.py
utility file : word_processing.py, helpers.py

Tensorflow Model file : model.py

-----------------------------------------------------
EXECUTE ORDER 

1. python training_1.py -r [F|T]
 >> learning word orde in sentence

2. python FinalStateClustering_2.py
 >> save final state, and cluster final state

3. python training_sequence_cluster_3.py -r [F|T]
 >> learning sequence order of cluster in each document

