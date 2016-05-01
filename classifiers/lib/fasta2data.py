import sys
from Bio import SeqIO

if len(sys.argv) == 1:
	print 'fasta2data.py [positive.fasta] [negative.fasta] [number_of_sequences]'
	sys.exit()

pos = sys.argv[1]
neg = sys.argv[2]
num = int(sys.argv[3])
counter = 0
for i in SeqIO.parse(pos, 'fasta'):
	i = str(i.seq)
	if counter < num:
		print '1',
		print i
		counter += 1
	else:
		break
counter = 0
for i in SeqIO.parse(neg, 'fasta'):
	i = str(i.seq)
	if counter < num:
		print '2',
		print i
		counter += 1
	else:
		break
