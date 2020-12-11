import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import BernoulliRBM
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection  import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import time


#show more column
pd.set_option('display.max_columns', 1000)
pd.set_option("display.max_colwidth",1000)

data = pd.read_csv('census.csv')
income_raw = data['income']
features_raw = data.drop('income', axis = 1)
skewed = ['capital-gain', 'capital-loss']
features_log_transformed = pd.DataFrame(data = features_raw)
features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))
scaler = MinMaxScaler() # default=(0, 1)
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])
features_final = pd.get_dummies(features_log_minmax_transform)
encoder = LabelEncoder()
income = encoder.fit_transform(income_raw)


X_train, X_test, y_train, y_test = train_test_split(features_final,
                                                    income,
                                                    test_size = 0.8,

                                                    random_state = 0)
start = time.time()
clf = MLPClassifier(solver='sgd',activation = 'identity',max_iter = 70,
                    alpha = 1e-5,hidden_layer_sizes = (100,50),random_state = 1,verbose = True)
clf.fit(X_train,y_train)
cost = time.time() - start
y_pred = clf.predict(X_test)
print('accuracy:',clf.score(X_test, y_test))
print('time:',cost)






# x = [i for i in range(1,100)]
# plt.plot(x, acc, marker='o', mec='r', mfc='w')
# plt.show()
# print(acc)
# print(time.time()-start)




# # ROC AUC
from sklearn import metrics
y_pred = clf.predict(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
# plt.plot(fpr,tpr,marker = 'o')
# plt.show()

from sklearn.metrics import auc
AUC = auc(fpr, tpr)
print(AUC)
#
# from sklearn.metrics import precision_score
# y_pred = clf.predict(X_test)
print('precision macro:',precision_score(y_test, y_pred, average='macro'))
print('precision micro:',precision_score(y_test, y_pred, average='micro'))
print('precision weighted:',precision_score(y_test, y_pred, average='weighted'))
print('precision 0-1:',precision_score(y_test, y_pred, average=None))


# from sklearn.metrics import recall_score
# y_pred = clf.predict(X_test)
print('recall macro:',recall_score(y_test, y_pred, average='macro'))
print('recall micro:',recall_score(y_test, y_pred, average='micro'))
print('recall weighted:',recall_score(y_test, y_pred, average='weighted'))
print('recall 0-1:',recall_score(y_test, y_pred, average=None))

# from sklearn.metrics import f1_score
# y_pred = clf.predict(X_test)
print('f1 macro:',f1_score(y_test, y_pred, average='macro'))
print('f1 micro:',f1_score(y_test, y_pred, average='micro'))
print('f1 weighted:',f1_score(y_test, y_pred, average='weighted'))
print('f1 0-1:',f1_score(y_test, y_pred, average=None))



from sklearn.metrics import confusion_matrix
y_pred = clf.predict(X_test)
print('confusion_matrix:\n',confusion_matrix(y_test, y_pred))














