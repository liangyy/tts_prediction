import sys
import math


single_bonds={
	'A': {'U': -4.42},
	'C': {'G': -5.53},
	'U': {'A': -4.42},
	'G': {'C': -5.53},
	# 'U': {'G':-4.45},
	# 'G': {'U':-4.45},
	# 'C': {'U': -0.37},
	# 'U': {'U': -5.82, 'C': -0.37},
}

helices_energy = {
	'AU': {'AU': -0.93, 'UA': -1.10, 'GC': -2.08, 'CG': -2.24, 'UG': -1.36, 'GU': -0.55},
	'UA': {'AU': -1.33, 'UA': -0.93, 'GC': -2.11, 'CG': -2.35, 'UG': -1.27, 'GU': -1.00},
	'CG': {'UA': -2.08, 'AU': -2.11, 'GC': -2.36, 'CG': -3.26, 'UG': -2.11, 'GU': -1.41},
	'GC': {'UA': -2.24, 'AU': -2.35, 'GC': -3.26, 'CG': -3.24, 'UG': -2.51, 'GU': -1.53},

	# GU pairs
	'GU': {'AU': -1.27, 'UA': -1.36, 'GC': -2.11, 'CG': -2.51, 'UG': 1.29, 'GU': -0.5}, # note a GU,GU (http://rna.urmc.rochester.edu/NNDB/turner04/gu-parameters.html)
	'UG': {'AU': -1.00, 'UA': -0.55, 'GC': -1.41, 'CG': -1.53, 'UG': -0.5, 'GU': 0.30}, # note a UG,UG

	# GGUC
	# CUGG energy is -1.77 but actually -4.12
}


hairpin_initiation = {
	3:5.4,
	4:5.6,
	5:5.7,
	6:5.4,
	7:6.0,
	8:5.5,
	9:6.4,
}


terminal_mismatch = {
	'AU': {
		'A': {'A': -0.8, 'C': -1.0, 'G': -0.8, 'U': -1.0},
		'C': {'A': -0.6, 'C': -0.7, 'G': -0.6, 'U': -0.7},
		'G': {'A': -0.8, 'C': -1.0, 'G': -0.8, 'U': -1.0},
		'U': {'A': -0.6, 'C': -0.8, 'G': -0.6, 'U': -0.8},
	},
	'CG': {
		'A': {'A': -1.5, 'C': -1.5, 'G': -1.4, 'U': -1.5},
		'C': {'A': -1.0, 'C': -1.1, 'G': -1.0, 'U': -0.8},
		'G': {'A': -1.4, 'C': -1.5, 'G': -1.6, 'U': -1.5},
		'U': {'A': -1.0, 'C': -1.4, 'G': -1.0, 'U': -1.2},
	},
	'GC': {
		'A': {'A': -1.1, 'C': -1.5, 'G': -1.3, 'U': -1.5},
		'C': {'A': -1.1, 'C': -0.7, 'G': -1.1, 'U': -0.5},
		'G': {'A': -1.6, 'C': -1.5, 'G': -1.4, 'U': -1.5},
		'U': {'A': -1.1, 'C': -1.0, 'G': -1.1, 'U': -0.7},
	},
	'GU': {
		'A': {'A': -0.3, 'C': -1.0, 'G': -0.8, 'U': -1.0},
		'C': {'A': -0.6, 'C': -0.7, 'G': -0.6, 'U': -0.7},
		'G': {'A': -0.6, 'C': -1.0, 'G': -0.8, 'U': -1.0},
		'U': {'A': -0.6, 'C': -0.8, 'G': -0.6, 'U': -0.6},
	},
	'UA': {
		'A': {'A': -1.0, 'C': -0.8, 'G': -1.1, 'U': -0.8},
		'C': {'A': -0.7, 'C': -0.6, 'G': -0.7, 'U': -0.5},
		'G': {'A': -1.1, 'C': -0.8, 'G': -1.2, 'U': -0.8},
		'U': {'A': -0.7, 'C': -0.6, 'G': -0.7, 'U': -0.5},
	},
	'UG': {
		'A': {'A': -1.0, 'C': -0.8, 'G': -1.1, 'U': -0.8},
		'C': {'A': -0.7, 'C': -0.6, 'G': -0.7, 'U': -0.5},
		'G': {'A': -0.5, 'C': -0.8, 'G': -0.8, 'U': -0.8},
		'U': {'A': -0.7, 'C': -0.6, 'G': -0.7, 'U': -0.5},
	},
}


special_hairpin_loops = {

	# Three nucleotides
	'CAACG': 6.8,
	'GUUAC': 6.9,

	# Four nucleotides
	'CUACGG': 2.8,
	'CUCCGG': 2.7,
	'CUUCGG': 3.7,
	'CUUUGG': 3.7,
	'CCAAGG': 3.3,
	'CCCAGG': 3.4,
	'CCGAGG': 3.5,
	'CCUAGG': 3.7,
	'CCACGG': 3.7,
	'CCGCGG': 3.6,
	'CCUCGG': 2.5,
	'CUAAGG': 3.6,
	'CUCAGG': 3.7,
	'CUUAGG': 3.5,
	'CUGCGG': 2.8,
	'CAACGG': 5.5,

	# Six nucleotides
 	'ACAGUGCU': 2.9,
 	'ACAGUGAU': 3.6,
 	'ACAGUGUU': 1.8,
 	'ACAGUACU': 2.8,
}



dangling_energy = {
	3: {
		'AU': {'A': -0.8, 'C': -0.5, 'G': -0.8, 'U': -0.6},
		'CG': {'A': -1.7, 'C': -0.8, 'G': -1.7, 'U': -1.2},
		'GC': {'A': -1.1, 'C': -0.4, 'G': -1.3, 'U': -0.6},
		'GU': {'A': -0.8, 'C': -0.5, 'G': -0.8, 'U': -0.6},
		'UA': {'A': -0.7, 'C': -0.1, 'G': -0.7, 'U': -0.1},
		'UG': {'A': -0.7, 'C': -0.1, 'G': -0.7, 'U': -0.1},
		'AG': {'A': 0.0, 'C': 0.0, 'G': 0.0, 'U': 0.0},
	},
	5: {
		'AU': {'A': -0.3, 'C': -0.1, 'G': -0.2, 'U': -0.2},
		'CG': {'A': -0.2, 'C': -0.3, 'G': -0.0, 'U': -0.0},
		'GC': {'A': -0.5, 'C': -0.3, 'G': -0.2, 'U': -0.1},
		'GU': {'A': -0.3, 'C': -0.1, 'G': -0.2, 'U': -0.2},
		'UA': {'A': -0.3, 'C': -0.3, 'G': -0.4, 'U': -0.2},
		'UG': {'A': -0.3, 'C': -0.3, 'G': -0.4, 'U': -0.2},
		'AG': {'A': 0.0, 'C': 0.0, 'G': 0.0, 'U': 0.0},
	}	
}




def helix_energy(helix, seqList):
	
	# Intermolecular initiation energy
	energy =4.09

	##print helix
	
	# Symmetry
	FiveToThree = ""
	ThreeToFive = ""
	
	for first in range(0, len(helix)-1):
		second = first + 1
		firstPair = seqList[helix[first][0]]+seqList[helix[first][1]]
		secondPair = seqList[helix[second][0]]+seqList[helix[second][1]]
		##print firstPair,secondPair

		FiveToThree = FiveToThree + seqList[helix[first][0]]
		ThreeToFive = ThreeToFive + seqList[helix[first][1]]

		# AU GU end penalty 
		if first == 0:
			if firstPair=="AU" or firstPair=="UA" or firstPair=="GU" or firstPair=="UG":
				energy = energy + 0.45
				##print energy
		if second == len(helix)-1:
			FiveToThree = FiveToThree + seqList[helix[second][0]]
			ThreeToFive = ThreeToFive + seqList[helix[second][1]]
			if secondPair=="AU" or secondPair=="UA" or secondPair=="GU" or secondPair=="UG":
				energy = energy + 0.45
				##print energy

		# Energy for two base pairs

		try:
			energy = energy + 4.0 * helices_energy[firstPair][secondPair]
		except:
			energy = float("inf")
			print "Invalid Structure helices_energy not found!"

		##print energy

	# Symmetry
	if FiveToThree == ThreeToFive[::-1]:
		energy = energy + 0.43

	# If GGUC is in the helix then remove the wrong energy calculated previously then add its actual energy
	#    CUGG
	pos1 = FiveToThree.find("GGUC")
	pos2 = ThreeToFive.find("CUGG")
	if pos1 != -1 and pos2 != -1:
		if pos1 == pos2:
			energy = energy - (-1.77) + (-4.12)

				
	
	##print FiveToThree
	##print ThreeToFive
	##print energy
	##print
	return energy





def hairpin_energy(hairpin):

	# Each hairpin loop in the format [AAAAAA, AA, AU, 6]
	# [nucleotides in the hairpin, first mismatch, end pair, number of nucleotides in the hairpin]

	##print
	##print hairpin

	hairpinEnergy = 0.0
	number = hairpin[3] 

	# The nearest neighbor rules prohibit hairpin loops with fewer than 3 nucleotides.
	if number < 3:
		# Assign infinite energy to the hairpinEnergy and because the entire sequence structure is not valid
		hairpinEnergy = hairpinEnergy + float("inf")
		print "Invalid Structure hairpin loop < 3!"
		return hairpinEnergy

	# Special cases
	fullHairpinLoop = hairpin[2][0] + hairpin[0] + hairpin[2][1]
	if special_hairpin_loops.has_key(fullHairpinLoop):
		hairpinEnergy = special_hairpin_loops[fullHairpinLoop]
		##print hairpinEnergy
		##print
		##return the energy for the special hairpin loop
		return hairpinEnergy	

	# If not a special case 

	# Add initiation energy to hairpinEnergy
	if number <= 9:
		hairpinEnergy = hairpinEnergy + hairpin_initiation[number]
	elif number > 9:
		# G(37 initiation (n>9)) = G(37 initiation (9)) + 1.75 RT ln(n/9), where R is the gas constant and T is the absolute temperature.
		hairpinEnergy = hairpinEnergy + hairpin_initiation[9] + 1.75 * 0.616  * math.log(number/9)

	if number == 3:

		# If there are three nucleotides in the hairpin loop,
		# do not receive a sequence-dependent first mismatch term.  
		# All C hairpin loops of three nucleotides receive a stability penalty
		allCPenalty = 0 
		if hairpin[0] == "CCC":
			allCPenalty = 1.5
		hairpinEnergy = hairpinEnergy + allCPenalty
		##print hairpinEnergy
		##print
		return hairpinEnergy
			
	# If number > 3
	# Add terminal mismatch to hairpinEnergy
	# hairpinEnergy = hairpinEnergy + terminal_mismatch[hairpin[2]][hairpin[1][0]][hairpin[1][1]]
	try:
		hairpinEnergy = hairpinEnergy + terminal_mismatch[hairpin[2]][hairpin[1][0]][hairpin[1][1]]
	except:
		hairpinEnergy = float("inf")
		print "Invalid terminal_mismatch not found!", hairpin[2], hairpin[1][0], hairpin[1][1]

	# If all C loops
	# Add all c loops penalty to hairpinEnergy
	if hairpin[0] == "C" * number:
		hairpinEnergy = hairpinEnergy + 0.3 * number + 1.6

	# Add energy of first mismatch to hairpinEnergy
	# UU or GA first mismatch
	if hairpin[1] == "UU" or hairpin[1] == "GA":
		hairpinEnergy = hairpinEnergy - 0.9
	# GG first mismatch
	if hairpin[1] == "GG":
		hairpinEnergy = hairpinEnergy - 0.8

	# Add special GU closure energy to hairpinEnergy
	if hairpin[2] == "GU":
		hairpinEnergy = hairpinEnergy - 2.2

	##print hairpinEnergy
	return hairpinEnergy






def energy(dot, seq):

	seqList = list(seq)
	dotList = list(dot)

	# Find the sub-structures of the RNA sequence
	#----------------------------------------------------------------------------------------------
	# Find places to start
	# Inner start looks like (...), meaning no structure is nested inside
	# Each inner start has one haripin loop
	# Outer start looks like )..(, meaning there is an outer structure that nests the two inner structure

	innerStart = []
	outerStart = []

	# pLB indicate the previous bracket is the left bracket "("
	# left indicates the position of the previous left bracket
	# pRB indicate the previous bracket is the left bracket ")"
	# right indicates the position of the previous left bracket

	pLB = False
	LB = -1
	pRB = False
	RB = -1

	for x in range(0, len(dot)):
		if (dotList[x] == "("):
			LB = x
			pLB = True
			if (pRB == True):
				##print RB, x
				##print dotList[RB], dotList[x]
				outerStart.append([RB,x])
			pRB =False

		if (dotList[x] == ")"):
			RB = x
			pRB = True
			if (pLB == True):
				##print LB, x
				##print dotList[LB], dotList[x]
				innerStart.append([LB,x])
			pLB = False

	
	##print "innerStart: ", innerStart
	##print "outerStart: ", outerStart


	#-----------------------------------------------------------------------------

	# Analyze the most basic structures from innerStart
	# Baisc strcuture has one and only one hairloop at one end and a base pair on the other end, 
	# and may connect with other basic structures to form multibranch loops
	# Baisc structure may also contain helices, bulge loops and internal loops.
	# [range, hairpin loop, 2 or more consecutive pairs(helices), single pairs, unpaired 5'end nucleotides, unpaired 3'end nucleotides]
	# unpaired nucleotides can be used to calculate <Bugle loops> and <internal loops>
	# 2 or more consecutive pairs can be used to calculate <helices> and <GU pairs>

	# <hairpin loops>
	# hairpins stores hairpin loops
	# each hairpin loop in the format [AAAAAA, AA, AU, 6]
	# [nucleotides in the hairpin, first mismatch, end, number of nucleotides in the hairpin]

	##print
	##print ">>>>>>>>>Finding the basic structures........."
	##print

	hairpins = []

	basicStructs = []

	pairs = []

	helices = []
	
	singlePairs = []
	
	unpair5end = []

	unpair3end = []

	
	for x in innerStart:

		# Hairpin part
		hpBases = ""
		for y in range(x[0]+1, x[1]):
			hpBases = hpBases + seqList[y]
		hairpin = [hpBases, seqList[x[0]+1]+seq[x[1]-1], seqList[x[0]]+seq[x[1]], len(hpBases)]
		hairpins.append(hairpin)
		
		
	for x in innerStart:

		# hairpin part
		hpBases = ""
		for y in range(x[0]+1, x[1]):
			hpBases = hpBases + seqList[y]
		hairpin = [hpBases, seqList[x[0]+1]+seq[x[1]-1], seqList[x[0]]+seq[x[1]], len(hpBases), [x[0],x[1]]]
		hairpins.append(hairpin)
		
		# Range part
		i = x[0]
		j = x[1]
		left = -1
		right = -1
		# gap is used to determine if the pairs are consecutive and no dot between them
		gap = False
		end = False
		while (i != -1 and j != len(dotList)):

			while (dotList[i] == "."):
				unpair5end.append(i)
				i = i-1
				if i == -1:
					end = True
					break
				gap = True

			while (dotList[j] == "."):
				unpair3end.append(j)
				j = j+1
				if j == len(dotList):
					end = True
					break
				gap = True

			if end: break
				
			if ( dotList[i] == ")" or dotList[j] == "(" ):
				break
			
			if gap == False:
				pairs.append([i,j])

			if gap == True:
				if len(pairs) == 1:
					singlePairs.append(pairs[0])
					pairs = []
				elif len(pairs) > 1:
					helices.append(pairs)
					pairs = []
				pairs.append([i,j])
				gap = False

			left = i		
			right = j	
			i = i-1
			j = j+1

		structRange = [left, right]

		# Clean unpaired nucleotides - Remove those out of the structure range
		unpair5end = [n for n in unpair5end if n >= left]
		unpair3end = [n for n in unpair3end if n <= right]

		# Add single pair or helice that reaches the end and are not added above
		if len(pairs) == 1:
			singlePairs.append(pairs[0])
		else:
			helices.append(pairs)
						
		
		basicStructs.append([structRange, hairpin, helices, singlePairs, unpair5end, unpair3end])



		##print
		##print "range: ", left, right
		##print "helices: ", helices
		##print "singlePairs: ", singlePairs
		##print "unpair5end: ", unpair5end
		##print "unpair3end: ", unpair3end
		##print "hairpin loop: ", hairpin
		##print


		pairs = []
		helices = []
		unpair5end = []
		unpair3end = []
		singlePairs = []
		

	#-------------------------------------------------------------------------------------------------
	# Analyze the higher order structure that nests the basic structures from outerStart

	# Merge the ranges of basic structures with outerstart

	rangeList = []
	for n in basicStructs:
		rangeList.append(n[0])
	for n in outerStart:
		rangeList.append(n)

	##print rangeList
	
	# Sort rangeList (by insertion sort)
	for i in range(1, len(rangeList)):
    		tmp = rangeList[i]
    		k = i
    		while (k > 0 and tmp[0] < rangeList[k - 1][0]):
       			rangeList[k] = rangeList[k - 1]
        		k = k - 1
    		rangeList[k] = tmp

	##print rangeList

	# Merging
	for i in range(0,len(rangeList)-1):
		if rangeList[i][1] == rangeList[i+1][0]:
			rangeList[i+1][0]=rangeList[i][0]
			rangeList[i]=0
	rangeList = [x for x in rangeList if x != 0]

	##print rangeList


	##print
	##print ">>>>>>>>>Finding outer structures............."
	##print

	# If there is only one range in the rangeList, then it means we have found the highest order structure
	# and that range in the rangeList represents the inner range of that structure

	# If there are more than one in the rangeList and a range in the rangeList is not complete / not in the form ( )
	# that means we need to find the outer structure for structures with the complete range ()
	# so that we can merge that range of that outer structure with the incomplete range 
	# to find the range of the even outer structure 

	# While there are more than one range in the rangeList,
	# extend the incomplete range one by one and and merge after each extension
	# until there is only one range in the rangList

	# Use outerStructs to store the outer structures
	# in the format [outer range, inner range, helices, single pairs, unpaired 5'end nucleotides, unpaired 3'end nucleotides]
	
	outerStructs = []

	if len(rangeList) == 1 and len(rangeList) == 0: 
		##print "No outer structures."
		pass	

	while (len(rangeList) != 1 and len(rangeList) != 0):

		# Find the index of the inner structures with complete range in the rangList
		index = -1
		for i in range(0, len(rangeList)):
			if (dotList[rangeList[i][0]]+dotList[rangeList[i][1]] == "()"):
				#print  rangeList[i][0], rangeList[i][1]
				index = i
				break

		# Extend one inner range to find the range of the outer structure

		pairs = []
		helices = []
		singlePairs = []	
		unpair5end = []
		unpair3end = []

		i = rangeList[index][0]-1
		j = rangeList[index][1]+1

		innerRange = [i, j]
		left = -1
		right = -1

		# gap is used to determine if the pairs are consecutive and there is no dot between them						
		gap = False
		endDot = False
		while (i != -1 and j != len(dotList)):

			while (dotList[i] == "."):
				unpair5end.append(i)
				i = i-1
				if i == -1:
					endDot = True
					break
				gap = True

			while (dotList[j] == "."):
				unpair3end.append(j)
				j = j+1
				if j == len(dotList):
					endDot = True
					break
				gap = True
			
			if endDot: break
				
			if ( dotList[i] == ")" or dotList[j] == "(" ):
				break
			
			if gap == False:
				pairs.append([i,j])

			if gap == True:
				if len(pairs) == 1:
					singlePairs.append(pairs[0])
					pairs = []
				elif len(pairs) > 1:
					helices.append(pairs)
					pairs = []
				pairs.append([i,j])
				gap = False

			left = i		
			right = j
				
			i = i-1
			j = j+1


		outerRange = [left, right]
		##print "outerRange:", left, right						


		# Clean unpaired nucleotides - remove those out of the outeRange
		unpair5end = [n for n in unpair5end if n >= left]
		unpair3end = [n for n in unpair3end if n <= right]

		# Add single pairs or helices that reaches the end and are not added above
		if len(pairs) == 1:
			singlePairs.append(pairs[0])
		else:
			helices.append(pairs)

		
		outerStructs.append([outerRange, innerRange, helices, singlePairs, unpair5end, unpair3end])


		# Update the rangeList and merge the ranges in the rangList	
		rangeList[index] = outerRange[:]

		##print rangeList

		for i in range(0,len(rangeList)-1):
			if rangeList[i][1] == rangeList[i+1][0]:
				rangeList[i+1][0]=rangeList[i][0]
				rangeList[i]=0
		rangeList = [x for x in rangeList if x != 0]

		##print rangeList


		
	for x in outerStructs: 
		##print "outerStructs: ", x
		pass
	##print


	##print ">>>>>>>>>Finding the highest structure that nests the whole sequence..."
	##print



	#---------------------------------------------------------
	# Now only one range in the rangeList find the most out structure if there is one
	
	pairs = []
	helices = []
	singlePairs = []	
	unpair5end = []
	unpair3end = []
	mainStruct = []

	if len(rangeList) == 0:
		mainStruct = []

	elif rangeList[0][0] == 0 or rangeList[0][1] == len(dotList)-1:
		# No out most structure, has two or more structures at the same level
		mainStruct = []
		##print "No out most structure"

	else:

		i = rangeList[0][0]-1
		j = rangeList[0][1]+1
	
		innerRange = [i, j]
		left = -1
		right = -1

		##print i, j

		# gap is used to determine if the pairs are consecutive and no dot between them
		gap = False
		endDot = False

		while (i != -1 and j != len(dotList)):

			while (dotList[i] == "."):
				unpair5end.append(i)
				i = i-1
				if i == -1:
					endDot = True
					break
				gap = True
	
			while (dotList[j] == "."):
				if endDot: break
				unpair3end.append(j)
				j = j+1
				if j == len(dotList):
					endDot = True
					break
				gap = True
				
			if endDot: break
			
			if gap == False:
				pairs.append([i,j])

			if gap == True:
				if len(pairs) == 1:
					singlePairs.append(pairs[0])
					pairs = []
				elif len(pairs) > 1:
					helices.append(pairs)
					pairs = []
				pairs.append([i,j])
				gap = False

			left = i		
			right = j
				
			i = i-1
			j = j+1

		outerRange = [left, right]
		##print left
		##print right

		# Clean unpaired nucleotides - remove those out of the outeRange
		unpair5end = [n for n in unpair5end if n >= left]
		unpair3end = [n for n in unpair3end if n <= right]

		# Add single pair or helice that reaches the end and are not added above
		if len(pairs) == 1:
			singlePairs.append(pairs[0])
		else:
			helices.append(pairs)


		mainStruct = [outerRange, innerRange, helices, singlePairs, unpair5end, unpair3end]
		if helices:
			if not helices[0] and not singlePairs:
		 		mainStruct = []
				##print "no out most structure"

		if mainStruct:
			pass
		else:
			outerRange = [0, len(dotList)-1]

	##print "mainStruct: ", mainStruct



	#-----------------------------------------------------------------------------------------------------------------
	# Analyze stuff outside the most out structure <dangling ends>

	# dangling5 stores dangling at the 5' end
	dangling5 = []
	# dangling3 stores dangling at the 3'end
	dangling3 = []

	endPair = []

	if mainStruct:
		endPair = mainStruct[0]
		if mainStruct[0][0] != 0:
			for i in range(0, mainStruct[0][0]):
				dangling5.append(i)
		if mainStruct[0][1] != len(dotList)-1:
			for i in range(mainStruct[0][1]+1, len(dotList)):
				dangling3.append(i)
	elif rangeList:
		endPair = rangeList[0]
		if rangeList[0][0] != 0:
			for i in range(0, rangeList[0][0]):
				dangling5.append(i)
		if rangeList[0][1] != len(dotList)-1:
			for i in range(rangeList[0][1]+1, len(dotList)):
				dangling3.append(i)


	##print "endPair: ", endPair
	##print "dangling5: ",dangling5
	##print "dangling3: ",dangling3	




	totalEnergy = 0.0

	#-----------------------------------------------------------------------------------------------------------------
	# Calculating energy for watson crick helices
	##print
	##print "Calculating energy for helices........"
	##print

	for basicStruct in basicStructs:
		for helix in basicStruct[2]:
			# Reverse helix to 5 to 3 order
			helix = helix[::-1]
			energy = helix_energy(helix, seqList)
			totalEnergy = totalEnergy + energy


	if outerStructs:
		for outerStruct in outerStructs:
			for helix in outerStruct[2]:
				# Reverse helix to 5 to 3 order
				helix = helix[::-1]
				energy = helix_energy(helix, seqList)
				totalEnergy = totalEnergy + energy


	if mainStruct:
		for helix in mainStruct[2]:
			# Reverse helix to 5 to 3 order
			helix = helix[::-1]
			energy = helix_energy(helix, seqList)
			totalEnergy = totalEnergy + energy



	#------------------------------------------------------------------------------
	# Calculating energy for hairpin loops
	##print
	##print "Calculating energy for hairpin loops......."
	
	for basicStruct in basicStructs:
		
		# hairpin loop is basicStruct[1]
		# Each hairpin loop in the format [AAAAAA, AA, AU, 6]
		# [nucleotides in the hairpin, first mismatch, end, number of nucleotides in the hairpin]

		hairpinEnergy = hairpin_energy(basicStruct[1])
		totalEnergy = totalEnergy + hairpinEnergy
		
		##print
		


	#------------------------------------------------------------------------------	
	# Calculating energy for Dangling ends
	##print
	##print "Calculating energy for dangling ends......."
	# http://rna.urmc.rochester.edu/NNDB/turner04/de-example.html

	# 3' dangling end
	if not dangling5 and len(dangling3) == 1:
		end = seqList[endPair[1]]+seqList[endPair[0]]
		try:
			danglingEnergy = dangling_energy[3][end][seqList[dangling3[0]]]
		except:
			danglingEnergy=0.0
		##print "danglingEnergy:", danglingEnergy
		##print
		totalEnergy = totalEnergy + danglingEnergy
		

	# 5' dangling end
	if not dangling3 and len(dangling5) == 1:

		end = seqList[endPair[1]]+seqList[endPair[0]]
		try:
			danglingEnergy = dangling_energy[5][end][seqList[dangling5[0]]]
		except:
			danglingEnergy=0.0
		##print "danglingEnergy:", danglingEnergy
		##print
		totalEnergy = totalEnergy + danglingEnergy


	#------------------------------------------------------------------------------
	# Return the calculated total energy
	return totalEnergy
			
	

# MAIN ----------------------------------------------------------------------------------


# index 0         1         2         3         4         5         6         7
# index 01234567890123456789012345678901234567890123456789012345678901234567890123456789
# dot = ".((((..(....).))))."
# seq = "GCACAAAACCCCUAUGUGU"

# dot = "((.((((..(....).))))..(.((...))..(...)..(...).)))..(...)."
# seq = "AAGCACAAAACCCCUAUGUGUUCGAACCCUUCCACCCUUAUGGGACGUUCGACCCUA"

# dot = "(((...)))"
# seq= "AAAGGGUUU"

# dot = "................................................"
# seq = "AAAAAAAAAAAAAAAAAAAAACGCGCGCGCGCGCGCGCGCGCGCGCGG"

# print
# print dot
# print seq
# print "sequence length:", len(dot)
# print
# e = energy(dot, seq)
# print "The total energy is:", e

# print "Example:", helix_energy([[0,15],[1,14],[2,13],[3,12],[4,11],[5,10],[6,9],[7,8]], "GGUCGUGUGCGUGGUC")


# print "Tests for hairpin loops ---------------------"
# [nucleotides in the hairpin, first mismatch, end, number of nucleotides in the hairpin]
# hairpin_energy(["CC", "CC", "AA", 2])
# hairpin_energy(["AAAAAA", "AA", "AU", 6]) # G=4.6 (http://rna.urmc.rochester.edu/NNDB/turner04/hairpin-example-1.html
# hairpin_energy(["GGAAG", "GG", "AU", 5]) # G=4.1 (http://rna.urmc.rochester.edu/NNDB/turner04/hairpin-example-2.html
# hairpin_energy(["CGAG", "CG", "CG", 4]) # G=3.5 (http://rna.urmc.rochester.edu/NNDB/turner04/hairpin-example-3.html
# hairpin_energy(["CCCCCC", "CC", "AU", 6]) # G=8.1 (http://rna.urmc.rochester.edu/NNDB/turner04/hairpin-example-4.html
# hairpin_energy(["GGAAG", "GG", "GU", 5]) # G=1.9 (http://rna.urmc.rochester.edu/NNDB/turner04/hairpin-example-5.html
# print

