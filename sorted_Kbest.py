import os
import sys
import pickle

sys.path.append("../tools/")
# Your python down;t have this path to perform opeartion, so append along with bin

from pprint import pprint

# For multi-stage operations we gonna use Pipeline
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier

with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

data_dict.pop('TOTAL')
data_dict.pop('LOCKHART EUGENE E')  # salary is NaN
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')

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

k = len(all_features_list) - 1  # except target variable poi

data = featureFormat(data_dict, all_features_list)
labels, features = targetFeatureSplit(data)

k_best = SelectKBest(k=k)
k_best.fit(features, labels)

unsorted_pair_list = zip(all_features_list[1:], k_best.scores_)

# sorting based on scores
sorted_pair_list = sorted(unsorted_pair_list, key=lambda x: x[1], reverse=True)

pprint(sorted_pair_list)
