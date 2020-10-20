import csv
import numpy
from pandas import DataFrame
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

#parameters for grid seaches
DT_params = [{'criterion': ['gini', 'entropy'],
         'max_depth': [10, None],
         'min_samples_split': [2, 5, 10],
         'min_samples_leaf': [1, 2, 10]}]

MLP_params = [{'hidden_layer_sizes':[(30,50,),(10,10,10,)],
               'activation': ['logistic','tanh','relu','identity'],
               'solver': ['adam','sgd'],

                }]

#user menu to select model number and dataset
model_num = 0
set_num = 0
out_file = ""
file_suffix = ""
while model_num == 0:
    model_num= int(input("Enter number to select learning model: \n1.GNB\n2.BaseDT\n3.BestDT\n4.Per\n5.BaseMLP\n6.BestMLP\n"))
    if model_num == 1 :
        classifier = GaussianNB()
        out_file = "GNB"
    elif model_num == 2:
        classifier = DecisionTreeClassifier(criterion='entropy')
        out_file = "Base_DT"
    elif model_num == 3:
        classifier = DecisionTreeClassifier()
        DT_gridsearch = GridSearchCV(classifier, DT_params, verbose=1, cv=5)
        out_file = "Best_DT"
    elif model_num == 4:
        classifier = Perceptron()
        out_file = "Per"
    elif model_num == 5:
        classifier = MLPClassifier(hidden_layer_sizes=(100, ), activation='logistic', solver='sgd')
        out_file = "Base_MLP"
    elif model_num == 6:
        classifier = MLPClassifier(learning_rate_init=0.005)
        MLP_gridsearch = GridSearchCV(classifier,MLP_params, n_jobs=-1, verbose=1, cv=3)
        out_file = "Best_MLP"
    else :
        model_num=int(input("Enter valid number (1-6):"))
print("Model Selected\n")

while set_num == 0:
    set_num = int(input("Enter number of dataset (1 or 2):\n"))
    if set_num == 1:
        file_suffix = "1.csv"
    elif set_num == 2:
        file_suffix = "2.csv"
    else :
        set_num=int(input("Enter valid number (1 or 2)"))
print("set selected\n")

#build out file string based on user selections
out_file = out_file + file_suffix


#Read Dataframe from csv file
print("\n -- Reading input file \n ")
training_set = pd.read_csv('C:\\Users\\domha\\PycharmProjects\\comp472\\assignment1\\Assig1-Dataset\\Assig1-Dataset\\train_'+file_suffix,
                      delimiter=',', header=None)
#Split dataset into instances and classes
X = training_set.iloc[:, :-1]
y = training_set.iloc[:, -1]


#Fit classifier to training data. For best fit models, apply gridsearch to find the best estimator.
if model_num == 3:
    DT_gridsearch.fit(X, y)
    print("Best params: {}".format(DT_gridsearch.best_params_))
    classifier = DecisionTreeClassifier(**DT_gridsearch.best_params_)
    classifier.fit(X,y)
elif model_num == 6:
    MLP_gridsearch.fit(X, y)
    print("Best params: {}".format(MLP_gridsearch.best_params_))
    classifier = MLPClassifier(**MLP_gridsearch.best_params_)
    classifier.fit(X,y)
else:
    classifier.fit(X,y)

#load test set
test_set = pd.read_csv('C:\\Users\\domha\\PycharmProjects\\comp472\\assignment1\\Assig1-Dataset\\Assig1-Dataset\\test_with_label_'+file_suffix,
                      delimiter=',', header=None)
X_test = test_set.iloc[:, :-1]
y_test = test_set.iloc[:, -1]

#predict values of validation samples
pred = classifier.predict(X_test)

#print classification report of test set
report = classification_report(y_test, pred, output_dict='true')
print("Classification report for classifier %s:\n" % (classifier))
for key, value in report.items():
    print(key, ' : ', value)
#print confusion matrix
cm = confusion_matrix(y_test, pred)
print("Confusion matrix:\n%s" % cm)


#plot distribution of the frequency of each class
plt.hist(y, bins=np.arange(y.min(), y.max()))
plt.show()


#cast objects that are to be outputted to objects of type DataFrame, so we can output to csv file using panda
pred_out = pd.DataFrame(pred)
cm_out = pd.DataFrame(cm)
report_out = pd.DataFrame(report)
pred_out.to_csv('C:\\Users\\domha\\PycharmProjects\\comp472\\assignment1\\Assig1-Dataset\\Assig1-Dataset\\'+out_file,header='none')
report_out.to_csv('C:\\Users\\domha\\PycharmProjects\\comp472\\assignment1\\Assig1-Dataset\\Assig1-Dataset\\'+out_file,index='false',mode='a',header='none')
cm_out.to_csv('C:\\Users\\domha\\PycharmProjects\\comp472\\assignment1\\Assig1-Dataset\\Assig1-Dataset\\'+out_file,index='false', mode='a',header='none')


