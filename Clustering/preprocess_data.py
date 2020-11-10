def load_data(file):
    with open(file, 'r') as f:
        data = f.read()
        lines = data.splitlines()
        x = []
        y = []
        for line in lines:
            l = line.split()
            temp = []
            for i in range(2):
                temp.append(float(l[i]))
            x.append(temp)
            y.append(int(l[2]) - 1)
    return x, y