#!/usr/bin/python

from Bio import SeqIO
#import lib_classifier
import pickle
import sys
from lib_classifier import load_instance

if sys.argv[1] == '--help':
	print('./tts_prediction [classifier] [genome] [motif_path]')
	print('motif path tells where your motif files locate')
	sys.exit()

classifier = sys.argv[1]
genome = sys.argv[2]
path = sys.argv[3]
classifier = load_instance(classifier, path)

window_size = classifier.window_size
print 'window_size = ', window_size
print(window_size)
for i in SeqIO.parse(genome, 'fasta'):
	#print(i)
	chromsome = i.name
	seq = str(i.seq)
	print 'chr = ', chromsome, '  from 1 to', len(seq) - window_size
	for scanner in range(len(seq) - window_size):
		subseq = seq[scanner : scanner + window_size]
		re = classifier.predict([subseq])
		#print(re[0])
		if re[0] == '1': # in training set 1 means TTS and other means non-TTS
			print scanner




