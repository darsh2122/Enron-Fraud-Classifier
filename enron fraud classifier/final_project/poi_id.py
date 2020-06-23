#!/usr/bin/python

import sys
from time import time
import pickle
import pandas as pd
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
#from tester import dump_classifier_and_data
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest,f_classif,chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from matplotlib import pyplot as plt

features_list = ["poi","salary","bonus","expenses","deferred_income","exercised_stock_options","total_payments","total_stock_value"]
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

print("people : ",len(data_dict))
print("no. of features :",len(data_dict['METTS MARK']))
poi_names = open("../final_project/poi_names.txt").read().split('\n')
poi_y = [name for name in poi_names if "(y)" in name]
poi_n = [name for name in poi_names if "(n)" in name]
print("poi in text: ",len(poi_y) + len(poi_n))
poi_count_database = 0
for i in data_dict:
	if data_dict[i]["poi"] == 1:
		poi_count_database += 1

print("poi in dataset : ",poi_count_database)

data_dict.pop('TOTAL',0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK',0)
### Task 3: Create new feature(s)
'''def getFormattedFeatures(data_dict, features):
    featureList = {}
    nameList = {}
    for feature in features:
        featureList[feature] = []
        nameList[feature] = []

    for feature in features:
        for name, value in data_dict.items():
            for key, val in value.items():
                if (key in features and val != 'NaN'):
                    featureList[key].append(val)
                    nameList[key].append(name)

    return featureList, nameList'''

def newFeature(data_dict,name,num,dem):
	for i in data_dict:
		if(data_dict[i][num] == "NaN" or data_dict[i][dem] == "NaN"):
			data_dict[i][name] = "NaN"
		else:
			data_dict[i][name] = float(data_dict[i][num])/float(data_dict[i][dem])
	features_list.append(name)
	return data_dict

def runClassifier(clf,features_train,features_test,labels_train,labels_test):
	t0 = time()
	clf.fit(features_train,labels_train)
	print("time to train the model : ",round(time()-t0,3),"s")
	t0 = time()
	clf.predict(features_test)
	accuracy.append(clf.score(features_test,labels_test))
	pre = metrics.precision_score(labels_test,clf.predict(features_test))
	re = metrics.recall_score(labels_test,clf.predict(features_test))
	f1s = metrics.f1_score(labels_test,clf.predict(features_test))
	precision.append(pre)
	recall.append(re)
	f1.append(f1s)
	print(clf.score(features_test,labels_test))
	print("time to predict test data : ",round(time()-t0,3),"s")

def findKbestFeatures(data_dict,features_list,k):
	data = featureFormat(data_dict,features_list)
	labels,features = targetFeatureSplit(data)
	k_best = SelectKBest(f_classif,k)#or chi2 can be used
	k_best.fit(features,labels)
	scores = k_best.scores_
	unsorted_pairs = zip(features_list[1:],scores)
	sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))	
	k_best_features = dict(sorted_pairs[:k])
	return k_best_features


data_dict = newFeature(data_dict,"salary_bonus", "salary","bonus")
data_dict = newFeature(data_dict,"salary_expenses","salary","expenses")
data_dict = newFeature(data_dict,"e_t","exercised_stock_options","total_stock_value")		
data_dict = newFeature(data_dict,"bonus_deferred","deferred_income","bonus")
data_dict = newFeature(data_dict,"t_s","total_payments","salary")
data_dict = newFeature(data_dict,"t_b","total_payments","bonus")
### Store to my_dataset for easy nexport below.
for i in data_dict:
	for j in data_dict[i]:
		if(data_dict[i][j] == "NaN"):
			data_dict[i][j] = 0
my_dataset = data_dict

selectKbestfeatures = findKbestFeatures(data_dict,features_list,k = 5)
print("Selected Features:",selectKbestfeatures)
features_list =['poi'] + list(selectKbestfeatures.keys())
### Extract features and labels from dataset for local testing
data = featureFormat(data_dict,features_list)
labels,features = targetFeatureSplit(data)

features_train,features_test,labels_train,labels_test = train_test_split(features,labels, test_size = 0.3,random_state=42)

accuracy =[]
precision = []
recall =[]
f1 = []

clf = GaussianNB()
runClassifier(clf,features_train,features_test,labels_train,labels_test)
clf = SVC(kernel = 'rbf' , C = 1000)
runClassifier(clf,features_train,features_test,labels_train,labels_test)
clf = DecisionTreeClassifier(min_samples_split = 8,criterion= "gini",max_depth=10)
runClassifier(clf,features_train,features_test,labels_train,labels_test)
clf = RandomForestClassifier(n_estimators = 5)
runClassifier(clf,features_train,features_test,labels_train,labels_test)
clf = AdaBoostClassifier(n_estimators= 1000)
runClassifier(clf,features_train,features_test,labels_train,labels_test)


print("Knn pipeline")
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
select = SelectKBest(score_func = chi2, k = 5)
pca = PCA(n_components = 5)
kneighs = KNeighborsClassifier(n_neighbors = 10, n_jobs = -1)
knn_steps = [('scaling', scaler),
        ('feature_selection', select),
        ('reduce_dim', pca),
        ('k_neighbors', kneighs)]

from sklearn.pipeline import Pipeline
kNN_pipeline = Pipeline(knn_steps)
t0 = time()
kNN_pipeline.fit(features_train, labels_train)
print("time for pipeline",round(t0 - time(),3),"s")
print(kNN_pipeline.score(features_test,labels_test))
barWidth = 0.25
r1 = np.arange(len(precision))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
 
# Make the plot
plt.bar(r1, precision, color='#7f6d5f', width=barWidth, edgecolor='white', label='precision')
plt.bar(r2, accuracy, color='#557f2d', width=barWidth, edgecolor='white', label='accuracy')
plt.bar(r3, recall, color='#2d7f5e', width=barWidth, edgecolor='white', label='recall')
plt.bar(r3, f1, color='#222f22', width=barWidth, edgecolor='white', label='f1')

plt.xlabel('group', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(precision))], ['GaussianNB', 'SVC', 'tree', 'RandomForestClassifier', 'AdaBoostClassifier'],rotation=90)
 
# Create legend & Show graphic
plt.legend()
plt.show()
#dump_classifier_and_data(clf, my_dataset, features_list)