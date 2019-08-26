import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

train_set_path = "/Users/weizhuozhang/Desktop/Files/train_encode_tuned.npy"
dev_set_path = "/Users/weizhuozhang/Desktop/Files/dev_encode_tuned.npy"
train_set_csv = "/Users/weizhuozhang/Desktop/Files/train_set_v1.txt"
dev_set_csv = "/Users/weizhuozhang/Desktop/Files/dev_set_v1.txt"

train_set = np.load(train_set_path)
train_csv = pd.read_csv(train_set_csv, sep='\t', header=None)
train_csv = np.array(train_csv)
dev_set = np.load(dev_set_path)
dev_csv = pd.read_csv(dev_set_csv, sep='\t', header=None)
dev_csv = np.array(dev_csv)

train_label = (train_csv[:,0]).astype('int')
dev_label = (dev_csv[:,0]).astype('int')

import time
k_neighbors = [3, 7, 9, 13, 15]
for k in k_neighbors:
    knn_clf = KNeighborsClassifier(n_jobs=-1, n_neighbors=k)
    knn_clf.fit(train_set, train_label)
    time_start=time.time()
    predicted = knn_clf.predict(dev_set)
    accuracy = sum(predicted == dev_label)/len(predicted)
    time_end=time.time()
    print("---------------------------------")
    print("knn k={0}".format(k))
    print("accuracy: {0}".format(accuracy))
    print("Time spent: {0:.2f}ms".format((time_end-time_start)*1000))
