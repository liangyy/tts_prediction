import sys
import re

def pssm(meme_filename):
	alength = 0
	width = 0
	probabilityMatrix = []

	with open(meme_filename, 'r') as memeF:
    		content = memeF.readlines()
		for line in content:
			probabilityMatrix.append(re.findall(r'\S+', line))

	width = len(probabilityMatrix)
	alength = len(probabilityMatrix[0])
				
	for i in range(0,width):
		for j in range(0,alength):
			probabilityMatrix[i][j] = float(probabilityMatrix[i][j])
		print probabilityMatrix[i]
	
	return probabilityMatrix
	



def scanSeq(sequence_filename, probabilityMatrix):

	width = len(probabilityMatrix)

	#for DNA the first column is A, the second is C, the third is G and the last is T

	sequence = ""
	with open(sequence_filename, 'r') as seqF:
    		content = seqF.readlines()
		for line in content:
			sequence = sequence + line.strip()
		sequence = list(sequence)
	print sequence


	maxScore = -float("inf")
	maxWindow = []
	maxPosition = []

	for position in range(0, len(sequence)-width+1):
		window = ""
		score = 1.0
		for index in range(0, width):
			window = window + sequence[position+index]
			nuc = -1
			if sequence[position+index] == "A":
				nuc = 0
			if sequence[position+index] == "C":
				nuc = 1
			if sequence[position+index] == "G":
				nuc = 2
			if sequence[position+index] == "T":
				nuc = 3
			score = score * probabilityMatrix[index][nuc]
		if score >= maxScore:
			if score > maxScore:
				maxWindow = []
				maxPosition = []
			maxScore = score
			maxWindow.append(window)
			maxPosition.append(position)

		print position+1, "  ", window, "  ", score


	print
	print "MAX:", score
	for i in range(0, len(maxWindow)):
		print maxPosition[i]+1, "  ", maxWindow[i]		

	


meme_filename = sys.argv[1]
sequence_filename = sys.argv[2]
print meme_filename
print sequence_filename

probabilityMatrix = pssm(meme_filename)
scanSeq(sequence_filename, probabilityMatrix)