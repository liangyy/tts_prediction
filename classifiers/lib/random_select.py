import sys
import subprocess
import random
from Bio import SeqIO

if len(sys.argv) == 1:
	print 'random_select.py [input_fasta] [num_of_sequences_to_select]'
	sys.exit()

fasta = sys.argv[1]
num = int(sys.argv[2])

p1 = subprocess.Popen(['cat', fasta], stdout=subprocess.PIPE)
# print(p1.communicate())

p2 = subprocess.Popen(['grep', '>'], stdin=p1.stdout, stdout=subprocess.PIPE)
#print(p2.communicate())
p1.stdout.close()

p3 = subprocess.Popen(['wc', '-l'], stdin=p2.stdout, stdout=subprocess.PIPE)
p2.stdout.close()
x = p3.communicate()[0]
x = int(x.strip())

thresold = float(num) / float(x)
for i in SeqIO.parse(fasta, 'fasta'):
	i = str(i.seq)
	r = random.random()
	if r < thresold:
		print '>selected_random'
		print i

