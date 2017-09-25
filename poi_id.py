#!/usr/bin/python

import sys
import pickle
import os

sys.path.append("../tools/")

import matplotlib.pyplot as plt

from sklearn import preprocessing
from pprint import pprint
from time import time
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.grid_search import GridSearchCV


from feature_format import featureFormat
from feature_format import targetFeatureSplit


# features_list is a list of strings, each of which is a feature name
# first feature must be "poi", as this will be singled out as the label

all_features_list = [  # person of interest
    'poi',
    # financial
    'salary', 'deferral_payments', 'total_payments', 'loan_advances',
    'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value',
    'expenses', 'exercised_stock_options', 'other', 'long_term_incentive',
    'restricted_stock', 'director_fees',
    # email
    'to_messages', 'from_poi_to_this_person', 'from_messages',
    'from_this_person_to_poi', 'shared_receipt_with_poi'
]

features_list = ["poi", "fraction_from_poi_email", "fraction_to_poi_email", 'shared_receipt_with_poi']

# load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r"))

# look at data

print type(data_dict)

# Size of Data
print "--------------"
print "No of datapoints(people) in dataset: ", len(data_dict.keys())
print "--------------"

# no of feature for each person
count = 0
for employee in data_dict:
    # 'data_dict[employee]' is itself a dictionary, loop through it
    for feature, feature_value in data_dict[employee].iteritems():
        # print feature
        count += 1
    break
print "For each person(key), {} features are available".format(count)
print "------------"

# pprint(data_dict['BUY RICHARD B'])
# print data_dict.values()

# Number of POI in data
POIcount = 0
for person in data_dict:
    if data_dict[person]['poi']:
        POIcount += 1
print "Total no of 'POI' identified persons: {}".format(POIcount)

nonPOIcount = [person for person in data_dict if data_dict[person]['poi'] == False]
print "Total no of 'nonPOI' identified persons: {}".format(len(nonPOIcount))
print 50 * "-"

# remove any outliers before proceeding further
features = ["salary", "bonus"]
data_dict.pop('TOTAL')
data_dict.pop('LOCKHART EUGENE E')  # salary is NaN
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')

data = featureFormat(data_dict, features)


# remove NAN's from dataset
outliers = []
for key in data_dict:
    val = data_dict[key]['salary']
    if val == 'NaN':
        continue
    outliers.append((key, int(val)))

outliers_final = (sorted(outliers, key=lambda x: x[1], reverse=True)[:10])
# uncomment for printing top 10 salaries
print "People having top 10 salaries:"
pprint(outliers_final)
print 50 * "-"


# plot features
for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter(salary, bonus)

plt.xlabel("salary")
plt.ylabel("bonus")
# uncomment to check outliers
# plt.show()


# create new features
# new features are: fraction_to_poi_email,fraction_from_poi_email

def dict_to_list(key, normalizer):
    new_list = []

    for i in data_dict:
        if data_dict[i][key] == "NaN" or data_dict[i][normalizer] == "NaN":
            new_list.append(0.)
        elif data_dict[i][key] >= 0:
            new_list.append(float(data_dict[i][key]) / float(data_dict[i][normalizer]))
    return new_list


# create two lists of new features
fraction_from_poi_email = dict_to_list("from_poi_to_this_person", "to_messages")
fraction_to_poi_email = dict_to_list("from_this_person_to_poi", "from_messages")

# insert new features into data_dict
count = 0
for i in data_dict:
    data_dict[i]["fraction_from_poi_email"] = fraction_from_poi_email[count]
    data_dict[i]["fraction_to_poi_email"] = fraction_to_poi_email[count]
    count += 1


# store to my_dataset for easy export below
my_dataset = data_dict


# these two lines extract the features specified in features_list
# and extract them from data_dict, returning a numpy array
data = featureFormat(my_dataset, features_list)

# plot new features
for point in data:
    from_poi = point[1]
    to_poi = point[2]
    plt.scatter(from_poi, to_poi)
    if point[0] == 1:
        plt.scatter(from_poi, to_poi, color="r", marker="*")
plt.xlabel("fraction of emails this person gets from poi")
# plt.show()


# if you are creating new features, could also do that here


# split into labels and features (this line assumes that the first
# feature in the array is the label, which is why "poi" must always
# be first in features_list
labels, features = targetFeatureSplit(data)


# machine learning goes here!
# please name your classifier clf for easy export below

# deploying feature selection
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.1, random_state=42)

# use KFold for split and validate algorithm
from sklearn.cross_validation import KFold
kf = KFold(len(labels), 3)
for train_indices, test_indices in kf:
    # make training and testing sets
    features_train = [features[ii] for ii in train_indices]
    features_test = [features[ii] for ii in test_indices]
    labels_train = [labels[ii] for ii in train_indices]
    labels_test = [labels[ii] for ii in test_indices]

from sklearn.tree import DecisionTreeClassifier

t0 = time()

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
score = clf.score(features_test, labels_test)
print 'DT accuracy before tuning ', score

print "DT algorithm time:", round(time() - t0, 3), "s"
print 50 * "-"


importances = clf.feature_importances_
import numpy as np
indices = np.argsort(importances)[::-1]
print 'Feature Ranking: '
for i in range(3):
    print "{} feature {} ({})".format(i + 1, features_list[i + 1], importances[indices[i]])
print 50 * "-"


# try Naive Bayes for prediction
t0 = time()

clf = GaussianNB()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy = accuracy_score(pred, labels_test)


print 'NB accuracy comparision:', accuracy

print "NB algorithm time:", round(time() - t0, 3), "s"
print 50 * "-"


# use manual tuning parameter min_samples_split

clf = DecisionTreeClassifier(min_samples_split=5)
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

acc = accuracy_score(labels_test, pred)

print "Validating algorithm:"
print "DT accuracy after tuning = ", acc

# function for calculation ratio of true positives
# out of all positives (true + false)
print 'DT precision = ', precision_score(labels_test, pred)

# function for calculation ratio of true positives
# out of true positives and false negatives
print 'DT recall = ', recall_score(labels_test, pred)


# dump the classifier, dataset and features_list
pickle.dump(clf, open("my_classifier.pkl", "w"))
pickle.dump(data_dict, open("my_dataset.pkl", "w"))
pickle.dump(features_list, open("my_feature_list.pkl", "w"))
