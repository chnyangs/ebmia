import statistics as st

import numpy as np

with open("results.txt", 'r') as f:
    lines = f.readlines()
    index = 1
    acc, pre, recall, f1, dis = [], [], [], [], []
    for line in lines:
        # print(index, line)
        if index % 5 == 0:
            values = line.strip().split(',')
            acc.append(float(values[0]))
            pre.append(float(values[1]))
            recall.append(float(values[2]))
            f1.append(float(values[3]))
            dis.append(float(values[4]))
        index += 1

    pre_mean = st.mean(pre)
    pre_std = st.stdev(pre)
    rec_mean = st.mean(recall)
    rec_std = st.stdev(recall)
    f1s_mean = st.mean(f1)
    f1s_std = st.stdev(f1)
    print("Mean:{},{},{}".format(round(pre_mean, 4), round(rec_mean, 4), round(f1s_mean, 4)))
    print("Std:{},{},{}".format(round(pre_std, 4), round(rec_std, 4), round(f1s_std, 4)))

    # for a in acc:
    #     print(a)
    # print()
    # for p in pre:
    #     print(p)
    # print()
    # for r in recall:
    #     print(r)
    # print()
    # for f in f1:
    #     print(f)
    # print()
    # for d in dis:
    #     print(d)
