with open("results.txt", 'r') as f:
    lines = f.readlines()
    index = 1
    acc, pre, recall, f1, dis = [], [], [], [], []
    for line in lines:
        if index % 5 == 0:
            values = line.strip().split(',')
            acc.append(values[0])
            pre.append(values[1])
            recall.append(values[2])
            f1.append(values[3])
            dis.append(values[4])
        index += 1

    for a in acc:
        print(a)
    print()
    for p in pre:
        print(p)
    print()
    for r in recall:
        print(r)
    print()
    for f in f1:
        print(f)
    print()
    for d in dis:
        print(d)
