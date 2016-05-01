import sys

if len(sys.argv) == 1:
	print 'data2fasta.py [data] [positive_tag] [negative_tag] [positive_out] [negative_out]'
	sys.exit()

data = sys.argv[1]
pos = sys.argv[2]
neg = sys.argv[3]
poso = sys.argv[4]
nego = sys.argv[5]

posoh = open(poso, 'w')
negoh = open(nego, 'w')

datah = open(data, 'r')
datah = datah.readlines()
for i in datah:
	l = i.split(' ')
	if l[0] == pos:
		posoh.write('>pos\n')
		posoh.write(l[1])
	elif l[0] == neg:
		negoh.write('>neg\n')
		negoh.write(l[1])

posoh.close()
negoh.close()