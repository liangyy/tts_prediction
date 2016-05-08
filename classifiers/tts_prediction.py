from __future__ import print_function
import sys
import re
# from pyfasta import Fasta
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def report(subseq, new_char, reverse_seq, chromsome, scanner, starter, flag):
	# print(subseq)
	re = classifier.predict([subseq])
	#print(re[0])
	if flag == '':
		reverse_seq = update_reverse_seq(reverse_seq, new_char)
	if re[0] == 1: # in training set 1 means TTS and other means non-TTS
		word = [str(chromsome), '\t', str(scanner + starter), '\t', str(scanner + starter), '\t'\
				, 'predicted_tts', '\t', 'NA', '\t', '+']
		eprint(''.join(word))
		out.write(''.join(word) + '\n')
	else:
		eprint(str(chromsome) + '\t' + str(scanner + starter) + '\tskip\t+')
	re = classifier.predict([reverse_seq])
	if re[0] == 1: # in training set 1 means TTS and other means non-TTS
		word = [str(chromsome), '\t', str(scanner + starter), '\t', str(scanner + starter), '\t'\
				, 'predicted_tts', '\t', 'NA', '\t', '-']
		eprint(''.join(word))
		out.write(''.join(word) + '\n')
	else:
		eprint(str(chromsome) + '\t' + str(scanner + starter) + '\tskip\t-')
	return reverse_seq

def noreport(new_char, reverse_seq):
	reverse_seq = update_reverse_seq(reverse_seq, new_char)
	return reverse_seq


from Bio import SeqIO
#import lib_classifier
import pickle
import sys
from lib_classifier import load_instance
from lib_classifier import reverse_complementary
from lib_classifier import update_reverse_seq
import re
if len(sys.argv) == 1:
	print
	print('Program for discovering TTSs in Maize Genome')
	print('tts_prediction is for predicting the sites in a input genome, please use tts_prediction_train to train the model')
	print
	print('USAGE:')
	print('python tts_prediction.py [classifier] [genome] [motif_path] [jump_size] [out]')
	print
	print('for example:')
	print('[classifier]: test_classifier')
	print('[genome]: ./genome/test_genome.fa')
	print('[motif_path]: ./motifs')
	print('[jump_size]: 100')
	print
	sys.exit()


if sys.argv[1] == '--help':
	print
	print('Program for discovering TTSs in Maize Genome')
	print('tts_prediction is for predicting the sites in a input genome, please use tts_prediction_train to train the model')
	print
	print('USAGE:')
	print('python tts_prediction.py [classifier] [genome] [motif_path] [jump_size] [out]')
	print
	print('for example:')
	print('[classifier]: test_classifier')
	print('[genome]: ./genome/test_genome.fa')
	print('[motif_path]: ./motifs')
	print('[jump_size]: 100')
	print
	sys.exit()

classifier = sys.argv[1]
genome = sys.argv[2]
path = sys.argv[3]
classifier = load_instance(classifier, path)
jump_size = int(sys.argv[4])
out = sys.argv[5]
out = open(out, 'w')
window_size = classifier.window_size
eprint('window_size = ', window_size)
eprint(window_size)
starter = window_size / 2 + 1

# f = Fasta(genome)
f = open(genome, 'r')
f = f.readlines()
start_flag = 0
seq = ''
jump_counter = 0
for l in f:
	# f.pop()
	# print(l)
	l = re.sub('\n', '', l)
	if len(l) == 0:
		continue
	 # else:
		# print(l)
	if l[0] == '>':
		chromsome = l.split(':')
		chromsome = chromsome[0][1:]
		scanner = 0
		start_flag = 1
	else:
		if start_flag == 1:
			if len(seq + l) < window_size:
				seq += l
				continue
			else:
				d = window_size - len(seq)
				seq += l[0:d]
				subseq = seq
				reverse_seq = reverse_complementary(subseq)
				# print('scanner', scanner)
				reverse_seq = report(subseq, '', reverse_seq, chromsome, scanner, starter, 'not')
				start_flag = 0
				tmp = l[d:]
				chars = list(tmp)
				for e in chars:
					subseq = subseq[1:] + e
					scanner += 1
					# print('e = ',e)
					jump_counter += 1
					if jump_counter == jump_size:
						reverse_seq = report(subseq, e, reverse_seq, chromsome, scanner, starter, '')
						jump_counter = 0
					else:
						reverse_seq = noreport(e, reverse_seq)
				# f.insert(0, tmp)
				# print(f[0])
				continue
		elif start_flag == 0:
			chars = list(l)
			for e in chars:
				subseq = subseq[1:] + e
				scanner += 1
				jump_counter += 1
				if jump_counter == jump_size:
					reverse_seq = report(subseq, e, reverse_seq, chromsome, scanner, starter, '')
					jump_counter = 0
				else:
					reverse_seq = noreport(e, reverse_seq)
				

# print(f.keys())
# for key in f.keys():
	# chromsome = key
	# seq = str(f[key])
# for i in SeqIO.parse(genome, 'fasta'):

	#print(i)
	# chromsome = i.name
	# seq = str(i.seq)
	#print 'chr = ', chromsome, '  from 1 to', len(seq) - window_size
	# reverse_seq = seq[0 : window_size]
	# reverse_seq = reverse_complementary(reverse_seq)
	# for scanner in range(len(seq) - window_size):
	# 	# subseq = seq[scanner : scanner + window_size]
	# 	print('1\t')
	# 	subseq = str(f[key][scanner : scanner + window_size])
	# 	print(subseq)
	# 	if 'N' in subseq:
	# 		continue
	# 	new_char = str(f[key][scanner + window_size - 1])
	# 	re = classifier.predict([subseq])
	# 	#print(re[0])
	# 	reverse_seq = update_reverse_seq(reverse_seq, new_char)
	# 	if re[0] == 1: # in training set 1 means TTS and other means non-TTS
	# 		word = [str(chromsome), '\t', str(scanner + starter), '\t', str(scanner + starter), '\t'\
	# 				, 'predicted_tts', '\t', 'NA', '\t', '+']
	# 		eprint(''.join(word))
	# 		out.write(''.join(word) + '\n')
	# 	else:
	# 		eprint(str(chromsome) + '\t' + str(scanner + starter) + '\tskip\t+')
	# 	re = classifier.predict([reverse_seq])
	# 	if re[0] == 1: # in training set 1 means TTS and other means non-TTS
	# 		word = [str(chromsome), '\t', str(scanner + starter), '\t', str(scanner + starter), '\t'\
	# 				, 'predicted_tts', '\t', 'NA', '\t', '-']
	# 		eprint(''.join(word))
	# 		out.write(''.join(word) + '\n')
	# 	else:
	# 		eprint(str(chromsome) + '\t' + str(scanner + starter) + '\tskip\t-')







