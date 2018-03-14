import csv
import numpy as np
import pickle
from sklearn.svm import SVC

testdata = []
traindata = []
trainlabel = []

with open('train.labels.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        trainlabel.append(row)
        
testdt = pickle.load(open("nntest.pkl", 'rb'))
traindt = pickle.load(open("nntrain.pkl", 'rb'))

trainlb = np.array(trainlabel[1:12001], dtype=float).flatten()
devlb = np.array(trainlabel[12001:], dtype=float).flatten()
devdt = traindt[12000:,:]
traindt = traindt[:12000, :]
print(testdt.shape)
print(traindt.shape)
print(trainlb.shape)
print(devdt.shape)
print(devlb.shape)

np.random.seed(5)

#Gamma = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9, 2, 3, 10, 30, 100]
#C = [1, 3, 10, 30, 100, 300]
#Gamma = [0.1, 1, 3]
#C = [1, 3, 10]
#Gamma = np.linspace(1.5, 2.0, num=5)
#C = np.linspace(20,50,num=10)
Gamma = [1/256, 10/256, 100/256]
C = [1, 5, 9, 11, 15]
res = []

for c in C:
    for gamma in Gamma:
        res_dict = dict()
        clf = SVC(kernel="rbf", gamma=gamma, C=c, probability=True)
        clf.fit(traindt, trainlb)
        pred_res = clf.predict(devdt)
        pred_prob_1 = clf.predict_proba(devdt)[:,1]
        err = np.mean(np.abs(pred_res - devlb))
        loss = np.sum(devlb * np.log(pred_prob_1) + (1-devlb) * np.log(1-pred_prob_1))
        print(gamma, c, err, loss)
        res_dict["gamma"] = gamma
        res_dict["c"] = gamma
        res_dict["acc"] = 1.0 - err
        res_dict["loss"] = loss
        res.append(res_dict)

pickle.dump(res, open("gridres.pkl", "wb"))

