def process_tic_tac_toe():
	with open('data/tic-tac-toe.txt', 'r') as f:
		data = f.read()
		lines = data.splitlines()
		x_data = []
		y_data = []
		for line in lines:
			l = line.split(',')
			x = []
			y = []
			for d in l:
				if d == 'x':
					x.append(5)
				elif d == 'o':
					x.append(-5)
				elif d == 'b':
					x.append(1)
				elif d == 'positive':
					y_data.append(1)
				else:
					y_data.append(-1)
			x_data.append(x)

	add_w0(x_data)
	return x_data, y_data


def process_sonar():
	with open('data/sonar.txt', 'r') as f:
		data = f.read()
		lines = data.splitlines()
		x_data = []
		y_data = []
		for line in lines:
			x = []
			for l in line.split(','):
				if l == 'R':
					y_data.append(1)
				elif l == 'M':
					y_data.append(-1)
				else:
					x.append(float(l))
			x_data.append(x)				

	add_w0(x_data)
	return x_data, y_data

def process_occupancy():
	with open('data/occupancy.txt') as f:
		data = f.read()
		lines = data.splitlines()
		x_data = []
		y_data = []
		i = 0
		for line in lines:
			if i != 0:
				x = []
				l = line.split(',')
				for j in range(len(l)):
					if j == len(l) - 1:
						if l[j] == '0':
							y_data.append(-1)
						else:
							y_data.append(1)
					elif j > 1:
						x.append(float(l[j]))
				x_data.append(x)

			i += 1

	add_w0(x_data)
	return x_data, y_data

def add_w0(x):
	'''
	Helper function that appends a 1 to all x data points as a dummy feature
	representing the bias term w0.
	'''
	for d in x:
		d.append(1)







