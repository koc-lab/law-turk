import pickle
import numpy as np
import random
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree
from sklearn import svm
from sklearn import ensemble
from sklearn import metrics
from sklearn import neighbors
from sklearn import dummy
import time


def prepare_data(name, label_dict):


    train_ratio = 0.7
    val_ratio = 0.15


    data_file = name + '_data_PCA.law'
    label_file = name + '_labels.law'

    with open(data_file, 'rb') as pickle_file:
        data_matrix = pickle.load(pickle_file)

    with open(label_file, 'rb') as pickle_file:
        labels = pickle.load(pickle_file)


    scaler = MinMaxScaler(feature_range=(-1,1))
    data_matrix = scaler.fit_transform(data_matrix)

    list_indices = []

    for i, lbl in enumerate(labels):
        if lbl in label_dict:
            list_indices.append(i)


    random.Random(4).shuffle(list_indices)
    #random.Random(13).shuffle(list_indices)

    new_length = len(list_indices)

    train_idx = math.floor(new_length * train_ratio)
    val_idx = math.floor(new_length * (train_ratio + val_ratio))

    train_indices = list_indices[0:train_idx]
    val_indices = list_indices[train_idx:val_idx]
    test_indices = list_indices[val_idx:]

    train_matrix = np.zeros((len(train_indices), data_matrix.shape[1]))
    val_matrix = np.zeros((len(val_indices), data_matrix.shape[1]))
    test_matrix = np.zeros((len(test_indices), data_matrix.shape[1]))

    train_labels = []
    val_labels = []
    test_labels = []

    for i, ind in enumerate(train_indices):
        train_matrix[i,:] = data_matrix[ind,:]
        train_labels.append(label_dict[labels[ind]])

    for i, ind in enumerate(val_indices):
        val_matrix[i,:] = data_matrix[ind,:]
        val_labels.append(label_dict[labels[ind]])

    for i, ind in enumerate(test_indices):
        test_matrix[i,:] = data_matrix[ind,:]
        test_labels.append(label_dict[labels[ind]])

    return train_matrix, train_labels, val_matrix, val_labels, test_matrix, test_labels


def run_model(court, model_name, mode):
    startTime_s = time.time()

    name = 'data/' + court + '/' + court

    if court == 'criminal' or court == 'civil' or court == 'administrative' or court == 'taxation':
        label_dict = {'RED' : 0, 'KABUL' : 1}
    else:
        label_dict = {' İhlal' : 0, ' İhlal Olmadığı' : 1}

    train_matrix, train_labels, val_matrix, val_labels, test_matrix, test_labels = prepare_data(name, label_dict)

    # print(train_matrix.shape)
    # print(val_matrix.shape)
    # print(test_matrix.shape)

    # print(len(train_labels) + len(val_labels) + len(test_labels))
    # for key in label_dict:
    #     print(key, ': ', (train_labels.count(label_dict[key]) + val_labels.count(label_dict[key]) + test_labels.count(label_dict[key])))
    # print('Training: ', len(train_labels))
    # print('Validation: ', len(val_labels))
    # print('Test: ', len(test_labels))

    best_score = 0.0

    print('Begin Classification')

    # print(val_labels)
    # print(test_labels)

    if model_name == 'Dummy':

        dum = dummy.DummyClassifier(strategy='stratified')
        dum.fit(train_matrix, train_labels)

        val_hat = dum.predict(val_matrix)
        test_hat = dum.predict(test_matrix)

    elif model_name == 'DT':

        for crt in ['entropy', 'gini']:
            for minsamp in [1,2,3]:

                print('d', end='', flush=True)

                dt = tree.DecisionTreeClassifier(class_weight='balanced', criterion=crt, min_samples_leaf=minsamp)
                dt.fit(train_matrix, train_labels)

                val_pred = dt.predict(val_matrix)
                
                if metrics.f1_score(val_labels, val_pred, average='macro') > best_score:
                    best_score = metrics.f1_score(val_labels, val_pred, average='macro')
                    val_hat = val_pred
                    test_hat = dt.predict(test_matrix)

    elif model_name == 'SVM':

        for c in [0.01, 0.02, 0.04, 0.07, 0.1, 0.2, 0.4, 0.7, 1.0, 2.0, 5.0, 10.0]:
            for kern in ['rbf', 'linear']:

                print('s', end='', flush='True')

                svc = svm.SVC(C= c, kernel=kern, class_weight='balanced')
                svc.fit(train_matrix, train_labels)

                val_pred = svc.predict(val_matrix)
                
                if metrics.f1_score(val_labels, val_pred, average='macro') > best_score:
                    best_score = metrics.f1_score(val_labels, val_pred, average='macro')
                    val_hat = val_pred
                    test_hat = svc.predict(test_matrix)

    elif model_name == 'RF':

        for n_est in [1, 3, 10, 30, 100]:
            for min_samp in [1,2,3]:

                print('r', end='', flush=True)
                
                rf = ensemble.RandomForestClassifier(n_estimators=n_est, class_weight='balanced', min_samples_leaf=min_samp) # n_esstimators 100 for BAM_hukuk
                rf.fit(train_matrix, train_labels)

                val_pred = rf.predict(val_matrix)
                
                if metrics.f1_score(val_labels, val_pred, average='macro') > best_score:
                    best_score = metrics.f1_score(val_labels, val_pred, average='macro')
                    val_hat = val_pred
                    test_hat = rf.predict(test_matrix)
    
    else:

        return



    print('Validation Results:')

    # print(metrics.confusion_matrix(val_labels, val_hat))
    print(metrics.accuracy_score(val_labels, val_hat))
    print(metrics.balanced_accuracy_score(val_labels, val_hat))
    print(metrics.f1_score(val_labels, val_hat, average='macro'))

    if mode == 'test':
        print('Test Results:')
        print('Accuracy: ', metrics.accuracy_score(test_labels, test_hat))
        print('Balanced Accuracy: ', metrics.balanced_accuracy_score(test_labels, test_hat))
        print('Macro F1', metrics.f1_score(test_labels, test_hat, average='macro'))
        print('Macro Precision', metrics.precision_score(test_labels, test_hat, average='macro'))
        print('Macro Recall', metrics.recall_score(test_labels, test_hat, average='macro'))

    print('Total time passed during execution (s): ' + str(time.time() - startTime_s))
