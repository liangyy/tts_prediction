

from Bio import SeqIO
#import lib_classifier
import pickle
import sys
from lib_classifier import load_instance
from lib_classifier import reverse_complementary
from lib_classifier import update_reverse_seq

if len(sys.argv) == 1:
	print
	print('Program for discovering TTSs in Maize Genome')
	print('tts_prediction is for predicting the sites in a input genome, please use tts_prediction_train to train the model')
	print
	print('USAGE:')
	print('python tts_prediction.py [classifier] [genome] [motif_path]')
	print
	print('for example:')
	print('[classifier]: test_classifier')
	print('[genome]: ./genome/test_genome.fa')
	print('[motif_path]: ./motifs')
	print
	sys.exit()


if sys.argv[1] == '--help':
	print
	print('Program for discovering TTSs in Maize Genome')
	print('tts_prediction is for predicting the sites in a input genome, please use tts_prediction_train to train the model')
	print
	print('USAGE:')
	print('python tts_prediction.py [classifier] [genome] [motif_path]')
	print
	print('for example:')
	print('[classifier]: test_classifier')
	print('[genome]: ./genome/test_genome.fa')
	print('[motif_path]: ./motifs')
	print
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
	#print 'chr = ', chromsome, '  from 1 to', len(seq) - window_size
	reverse_seq = seq[0 : window_size]
	reverse_seq = reverse_complementary(reverse_seq)
	for scanner in range(len(seq) - window_size):
		subseq = seq[scanner : scanner + window_size]
		new_char = seq[scanner + window_size - 1]
		re = classifier.predict([subseq])
		#print(re[0])
		reverse_seq = update_reverse_seq(reverse_seq, new_char)
		if re[0] == '1': # in training set 1 means TTS and other means non-TTS
			word = [str(chromsome), '\t', str(scanner), '\t', str(scanner), '\t'\
					, 'predicted_tts', '\t', 'NA', '\t', '+']
			print ''.join(word)
		re = classifier.predict([reverse_seq])
		if re[0] == '1': # in training set 1 means TTS and other means non-TTS
			word = [str(chromsome), '\t', str(scanner), '\t', str(scanner), '\t'\
					, 'predicted_tts', '\t', 'NA', '\t', '-']
			print ''.join(word)







