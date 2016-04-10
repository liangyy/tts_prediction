#!/usr/bin/env python

import sys
import math
import random

print


#score calculator
def score(B1, B2, i, j):
	if abs(j-i) <= 4:
		return 0
	if (B1 == "C" and B2 == "G"):
		return 1
	if (B1 == "G" and B2 == "C"):
		return 1
	if (B1 == "A" and B2 == “T”):
		return 1
	if (B1 == “T” and B2 == "A"):
		return 1
	return 0


#Input sequence
#sequence = sys.argv[1]
sequence='AUCGGCUAU'


length = len(sequence)

# Create matrix
matrix = state = [[[] for _ in range(length)] for _ in range(length)]


# for i in matrix:
# 	print i
# print


# Initialization
matrix[0][0] = 0
for i in range(1, length):
	matrix[i][i] = 0
	matrix[i][i-1] = 0


# for i in matrix:
# 	print i
# print


# Recursion
for x in range(1, length):
	for i in range(length):
		j = i + x
		if (i<length and j<length):
			maxOfTheThreeCells = max(matrix[i+1][j],matrix[i][j-1], matrix[i+1][j-1]+score(sequence[i], sequence[j], i, j))
			if j > i+1:
				bifurcation = []
				for k in range(i+1, j):
					bifurcation.append(matrix[i][k]+matrix[k+1][j])
				matrix[i][j] = max(maxOfTheThreeCells, max(bifurcation))
			else:
				matrix[i][j] = maxOfTheThreeCells


# for m in matrix:
# 	print m
# print


# Traceback
stack = []
i = 0
j = length -1
stack.append([i,j])
dot=[]
basepairs=[]
for i in xrange(length):
	dot.append('.')

while not stack==[]:

	pair = stack.pop()
	i = pair[0]
	j = pair[1]

	if (i >= j):
		continue

	elif matrix[i+1][j] == matrix[i][j]:
		stack.append([i+1, j])

	elif matrix[i][j-1] == matrix[i][j]:
		stack.append([i, j-1])

	elif (matrix[i+1][j-1] + score(sequence[i], sequence[j], i, j)) == matrix[i][j]:
		stack.append([i+1, j-1])
		basepairs.append((i,j))


		
	else:
		if (j > i+1):
			for k in range(i+1, j):
				if matrix[i][k]+matrix[k+1][j] == matrix[i][j]:
					stack.append([i, k])
					stack.append([k+1, j])
					break


for i in basepairs:
	k=i[0]
	j=i[1]
	dot[k]='('
	dot[j]=')'


print "".join(dot)

	

# Reference:
# http://www.tutorialspoint.com/python/list_max.htm
