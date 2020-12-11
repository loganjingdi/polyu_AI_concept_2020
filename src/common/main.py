'''
Using census.csv for evaluation
'''
# Import libraries necessary for this project
import numpy as np
import pandas as pd
from time import time
from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualization code visuals.py
import visuals as vs

# Import sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score,accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pickle


# Pretty display for notebooks
#%matplotlib inline


def display_data():
    # Success - Display the first record
    display(data.head(n=3))


# display_data()


def show_income():
    # TODO: Total number of records
    n_records = data.shape[0]

    # TODO: Number of records where individual's income is more than $50,000
    #n_greater_50k = sum(data['income'] == '>50K')

    # TODO: Number of records where individual's income is at most $50,000
    #n_at_most_50k = sum(data['income'] == '<=50K')
    n_at_most_50k, n_greater_50k = data.income.value_counts()

    # TODO: Percentage of individuals whose income is more than $50,000
    greater_percent = np.true_divide(n_greater_50k , n_records) * 100

    # Print the results
    print("Total number of records: {}".format(n_records))
    print("Individuals making more than $50,000: {}".format(n_greater_50k))
    print("Individuals making at most $50,000: {}".format(n_at_most_50k))
    print("Percentage of individuals making more than $50,000: {:.2f}%".format(greater_percent))


#show_income()

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test):
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''

    results = {}

    # TODO: Fit the learner to the training data using slicing with 'sample_size' using .fit(training_features[:], training_labels[:])
    start = time() # Get start time
    learner = learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time() # Get end time

    # TODO: Calculate the training time
    results['train_time'] = end - start
    print('training time:' + str(results['train_time']) + ' second(s)')

    # TODO: Get the predictions on the test set(X_test),
    #       then get predictions on the first 300 training samples(X_train) using .predict()
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time() # Get end time

    # TODO: Calculate the total prediction time
    results['pred_time'] = end-start
    print('testing time:' + str(results['pred_time']) + ' second(s)')

    # TODO: Compute accuracy on the first 300 training samples which is y_train[:300]
    results['acc_train'] = accuracy_score(predictions_train, y_train[:300])
    print('training accuracy:' + str(results['acc_train']))

    # TODO: Compute accuracy on test set using accuracy_score()
    results['acc_test'] = accuracy_score(predictions_test, y_test)
    print('testing accuracy:' + str(results['acc_test']))

    # TODO: Compute F-score on the the first 300 training samples using fbeta_score()
    results['f_train'] = fbeta_score(y_train[:300], predictions_train, beta= 0.5)
    print('training f-score:' + str(results['f_train']))

    # TODO: Compute F-score on the test set which is y_test
    results['f_test'] = fbeta_score(y_test, predictions_test, beta= 0.5)
    print('testing f-score:' + str(results['f_test']))

    # Success
    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))
    print("\n")
    # Return the results
    return results


def storeTree(inputTree, filename):
    fw = open(filename,'wb')
    pickle.dump(inputTree,fw)
    fw.close()


def grabTree(filename):
    fr = open(filename,'rb')
    return pickle.load(fr)


def evaluate(dataset, key_field):
    income_raw = dataset[key_field]

    # 1. start of pre-processing data
    features_raw = dataset.drop(key_field, axis = 1)

    # Visualize skewed continuous features of original data
    # vs.distribution(data)


    # Log-transform the skewed features
    skewed = ['capital-gain', 'capital-loss']
    features_log_transformed = pd.DataFrame(data = features_raw)
    features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))

    # Visualize the new log distributions
    # vs.distribution(features_log_transformed, transformed = True)


    # Initialize a scaler, then apply it to the features
    scaler = MinMaxScaler() # default=(0, 1)
    numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

    features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
    features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])

    # Show an example of a record with scaling applied
    display(features_log_minmax_transform.head(n = 5))

    # vs.distribution(features_log_minmax_transform)


    # TODO: One-hot encode the 'features_log_minmax_transform' data using pandas.get_dummies()
    features_final = pd.get_dummies(features_log_minmax_transform)

    # TODO: Encode the 'income_raw' data to numerical values
    encoder = LabelEncoder()
    income = encoder.fit_transform(income_raw)

    # Print the number of features after one-hot encoding
    encoded = list(features_final.columns)
    print("{} total features after one-hot encoding.".format(len(encoded)))

    # Uncomment the following line to see the encoded feature names
    # print encoded


    X_train, X_test, y_train, y_test = train_test_split(features_final,
                                                        income,
                                                        test_size = 0.2,
                                                        random_state = 0)

    # Show the results of the split
    print("Training set has {} samples.".format(X_train.shape[0]))
    print("Testing set has {} samples.".format(X_test.shape[0]))
    # 1. end of pre-processing data


    # 2. start of building native predictor
    '''
    TP = np.sum(income) # Counting the ones as this is the naive case. Note that 'income' is the 'income_raw' data
    encoded to numerical values done in the data preprocessing step.
    FP = income.count() - TP # Specific to the naive case

    TN = 0 # No predicted negatives in the naive case
    FN = 0 # No predicted negatives in the naive case
    '''
    # TODO: Calculate accuracy, precision and recall
    encoder = LabelEncoder()
    income = encoder.fit_transform(income_raw)

    TP = np.sum(income)
    FP = len(income) - TP

    accuracy = np.true_divide(TP,TP + FP)
    recall = 1
    precision = accuracy

    # TODO: Calculate F-score using the formula above for beta = 0.5 and correct values for precision and recall.
    # HINT: The formula above can be written as (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    fscore = (1 + 0.5**2) * (precision * recall) / ((0.5**2 * precision) + recall)

    # Print the results
    print("Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore))
    # 2. end of building native predictor


    # 3. start of evaluation
    # TODO: Initialize the three models
    clf_random_forest = RandomForestClassifier(max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=50)
    clf_decision_tree = DecisionTreeClassifier(random_state=0, max_depth=10, min_samples_leaf=6, min_samples_split=10)
    clf_C = SVC(kernel = 'rbf', C = 0.9, gamma = 0.9)
    clf_M = MLPClassifier(solver='sgd',activation = 'identity',max_iter = 70,alpha = 1e-5,hidden_layer_sizes = (100,50),random_state = 1,verbose = False)

    # TODO: Calculate the number of samples for 1%, 10%, and 100% of the training data
    # HINT: samples_100 is the entire training set i.e. len(y_train)
    # HINT: samples_10 is 10% of samples_100
    # HINT: samples_1 is 1% of samples_100
    samples_100 = len(y_train)
    samples_10 = int(len(y_train)*0.1)
    samples_1 = int(len(y_train)*0.01)

    # Collect results on the learners
    results = {}
    for clf in [clf_random_forest, clf_decision_tree, clf_C, clf_M]:
        clf_name = clf.__class__.__name__
        results[clf_name] = {}
        for i, samples in enumerate([samples_1, samples_10, samples_100]):
            results[clf_name][i] = train_predict(clf, samples, X_train, y_train, X_test, y_test)
            if clf == clf_decision_tree:
                storeTree(clf, "decision_tree")


    # Run metrics visualization for the three supervised learning models chosen
    vs.evaluate(results, accuracy, fscore)
    # 3. end of evaluation


dataset_map = {
    '1': {'filename': 'census.csv', 'key_field': 'income'},
    '2': {'filename': 'train.csv', 'key_field': 'exceeds50K'},
}


if __name__ == '__main__':
    dataset_info = dataset_map.get('1')
    data = pd.read_csv(dataset_info.get('filename'))
    evaluate(data, dataset_info.get('key_field'))