import sys
import re
if sys.argv[1] == '--help':
	print 'foo.py [dirty_input]'
	sys.exit()

dirty = sys.argv[1]
dh = open(dirty, 'r')
l = dh.readlines()
for i in l:
	i = re.sub('\n', '', i)
	e = i.split(' ')
	if 'N' in i:
		continue
	if len(e) < 2:
		continue
	else:
		if e[1].strip() == '':
			continue
		else:
			print i