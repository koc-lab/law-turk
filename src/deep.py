### Code for preparing data for the deep learning models, creating the models and training/testing
import time
import pickle
import numpy as np
import random
import math

# import nlp_utils
from gensim.models import KeyedVectors


from tensorflow.keras import layers
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras import utils
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU
from tensorflow.keras.layers import Embedding
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.utils import class_weight
from sklearn import metrics
from sklearn.decomposition import PCA

def pad_sequence(seq, max_len, one_hot):
    padded = [0 for i in range(max_len)]

    for i, word in enumerate(seq):
        if i < max_len:
            if word in one_hot:
                padded[i] = one_hot[word]

    return padded


def prepare_data(court_name, label_dict, max_len):

    
    ### Load data
    word_vectors = KeyedVectors.load_word2vec_format('data/trmodel', binary=True)

    with open('data/' + court_name + '/deep/' + court_name + '_tokenized.law','rb') as pickle_file:
        tokenized = pickle.load(pickle_file)
    with open('data/' + court_name + '/deep/' + court_name + '_labels.law','rb') as pickle_file:
        labels = pickle.load(pickle_file)

    

    ### vector_dict stores word embedding vectors that correspond to each word
    ### one_hot stores the index of each word in the embedding matrix

    vector_dict = {}
    one_hot = {}

    one_hot_index = 1

    tokenized_lower = []

    for text in tokenized:
        temp_text = []
        for word in text:
            temp_text.append(word.lower())
        # print(temp_text)
        tokenized_lower.append(temp_text)

    for text in tokenized_lower:
        for word in text:
            try:
                vector_dict[word] = (word_vectors.get_vector(word))
                if not word in one_hot:
                    one_hot[word] = one_hot_index
                    one_hot_index += 1
            except:
                pass

    vocab_size = len(one_hot)



    ### Split the data

    train_ratio = 0.70
    val_ratio = 0.15

    list_indices = []

    for i, lbl in enumerate(labels):
        if lbl in label_dict:
            list_indices.append(i)

    random.Random(13).shuffle(list_indices)

    new_length = len(list_indices)

    train_idx = math.floor(new_length * train_ratio)
    val_idx = math.floor(new_length * (train_ratio + val_ratio))

    train_indices = list_indices[0:train_idx]
    val_indices = list_indices[train_idx : val_idx]
    test_indices = list_indices[val_idx:]

    train_list = []
    val_list = []
    test_list = []

    for ind in train_indices:
        train_list.append(tokenized_lower[ind])

    for ind in val_indices:
        val_list.append(tokenized_lower[ind])

    for ind in test_indices:
        test_list.append(tokenized_lower[ind])

    train_labels = []
    val_labels = []
    test_labels = []

    for ind in train_indices:
        # train_labels.append(label_dict[labels[ind]])
        # train_labels.append(label_dict[labels[ind]])
        # train_labels.append(label_dict[labels[ind]])
        pass

    for ind in val_indices:
        val_labels.append(label_dict[labels[ind]])

    for ind in test_indices:
        test_labels.append(label_dict[labels[ind]])

    no_train = len(train_list)
    no_val = len(val_list)
    no_test = len(test_list)

    

    ### dim: embedding dimension
    wordsOfEmbedding = list(word_vectors.vocab.keys())
    dim = len(word_vectors.get_vector(wordsOfEmbedding[0]))

    train_data = []
    val_data = []
    test_data = []

    for i, text in enumerate(train_list):
        #print(text)
        padded = [0 for i in range(max_len)]
        max_ind = 0
        ind = 0
        for j, word in enumerate(text):
            if j < max_len:
                ind = ind + 1
                if word in one_hot:
                    max_ind = ind

        padded = pad_sequence(text, max_len, one_hot)

        train_data.append(padded)
        train_labels.append(label_dict[labels[train_indices[i]]])


        ### Code for chunking (augmenting) data, comment out for standard data

        chunk_size = 100

        for j in range(math.floor(max_ind / chunk_size)):
            start_ind = j * chunk_size
            end_ind = (j + 1) * chunk_size
            if end_ind < max_len - 1:
                padded_temp = pad_sequence(text[start_ind:end_ind], max_len, one_hot)
                train_data.append(padded_temp)
                train_labels.append(label_dict[labels[train_indices[i]]])

        ### End of chunking code
        


    for i, text in enumerate(val_list):
        padded = [0 for i in range(max_len)]
        for j, word in enumerate(text):
            if j < max_len:
                if word in one_hot:
                    padded[j] = one_hot[word]
        val_data.append(padded)
    
    for i, text in enumerate(test_list):
        padded = [0 for i in range(max_len)]
        for j, word in enumerate(text):
            if j < max_len:
                if word in one_hot:
                    padded[j] = one_hot[word]
        test_data.append(padded)



    ### Creation of an embedding matrix from individual embedding vectors

    embedding_matrix = np.zeros((vocab_size + 1, dim))

    for key, vec in vector_dict.items():
        embedding_matrix[one_hot[key], :] = vec



    ### Compute class weights

    clweights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
    
    clweights_dict = dict(enumerate(clweights))
    #clweights_dict = clweights
    
    return train_data, val_data, test_data, embedding_matrix, train_labels, val_labels, test_labels, clweights_dict, train_list, val_list, test_list






def run_model(court, model_name, mode, use_attention=True):

    court_name = court

    if mode == 'training':
        load_from_check = False
        train = True
        print_test_results = False
        startTime_s = time.time()
    elif mode == 'test':
        load_from_check = True
        train = False
        print_test_results = True
    else:
        return


    label_dict = {}
    if court == 'constitutional':
        label_dict = {' İhlal' : 0, ' İhlal Olmadığı' : 1}
    else:
        label_dict = {'RED' : 0, 'KABUL' : 1}
    

    no_classes = len(label_dict)


    ### max_len defines the clipping length for word sequences

    max_len = 1024
    
    train_data, val_data, test_data, embedding_matrix, train_labels, val_labels, test_labels, clweights, train_list, val_list, test_list = prepare_data(court_name, label_dict, max_len)
    dim = embedding_matrix.shape[1]
    # print(dim)


    ### Turn into numpy array

    train_data = np.array(train_data, dtype='int')
    val_data = np.array(val_data, dtype='int')
    test_data = np.array(test_data, dtype='int')



    #### One-hot encode labels

    train_labels_categorical = utils.to_categorical(train_labels)
    val_labels_categorical = utils.to_categorical(val_labels)
    test_labels_categorical = utils.to_categorical(test_labels)



    #checkpath = court_name + "weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
    if use_attention:
        checkpath = 'models/' + court_name + '_' + model_name + '_att.hdf5'
    else:
        checkpath = 'models/' + court_name + '_' + model_name + '.hdf5'
    
    checkpoint = ModelCheckpoint(checkpath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]




    load = False
    load_check = load_from_check
    train_more = train

    no_epochs = 30
    batch_size = 256


    # if load:
    #     model = load_model(mahkeme + '_model.h5')
    # else:
    if True:

        inputs = Input(shape=(max_len, ))
        embed = Embedding(embedding_matrix.shape[0], dim, weights=[embedding_matrix], input_length=max_len, trainable=False, mask_zero=True)(inputs)

        if model_name == 'GRU':

            if use_attention:
                x = GRU(dim, return_sequences=True)(embed)
                attention = TimeDistributed(Dense(1, activation='tanh'))(x) 
                attention = Flatten()(attention)
                attention = Activation('softmax')(attention)
                attention = RepeatVector(dim)(attention)
                attention = Permute([2, 1])(attention)
                sent_representation = Multiply()([x, attention])
                sent_representation = Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)
                probabilities = Dense(no_classes, activation='softmax')(sent_representation)
            else:
                x = GRU(dim, return_sequences=False)(embed)
                probabilities = Dense(no_classes, activation='softmax')(x)

        elif model_name == 'LSTM':

            if use_attention:
                x = LSTM(dim, return_sequences=True)(embed)
                attention = TimeDistributed(Dense(1, activation='tanh'))(x) 
                attention = Flatten()(attention)
                attention = Activation('softmax')(attention)
                attention = RepeatVector(dim)(attention)
                attention = Permute([2, 1])(attention)
                sent_representation = Multiply()([x, attention])
                sent_representation = Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)
                probabilities = Dense(no_classes, activation='softmax')(sent_representation)

            else:
                x = LSTM(dim, return_sequences=False, unit_forget_bias=True)(embed)
                probabilities = Dense(no_classes, activation='softmax')(x)

        elif model_name == 'BiLSTM':

            if use_attention:
                x = Bidirectional(LSTM(dim,  return_sequences=True, unit_forget_bias=True))(embed)
                attention = TimeDistributed(Dense(1, activation='tanh'))(x)
                attention = Flatten()(attention)
                attention = Activation('softmax')(attention)
                attention = RepeatVector(2 * dim)(attention)
                attention = Permute([2, 1])(attention)
                sent_representation = Multiply()([x, attention])
                sent_representation = Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)
                probabilities = Dense(no_classes, activation='softmax')(sent_representation)

            else:
                x = Bidirectional(LSTM(dim,  return_sequences=False, unit_forget_bias=True))(embed)
                probabilities = Dense(no_classes, activation='softmax')(x)

        else:
            
            return


        model = Model(inputs=[inputs], outputs=[probabilities])


        if load_check:
            model.load_weights(checkpath)

        model.summary()

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    if train_more:

        model.fit(train_data, train_labels_categorical, validation_data=(val_data, val_labels_categorical), batch_size=batch_size, epochs=no_epochs, class_weight=clweights, callbacks=callbacks_list)


    if print_test_results:

        print('Test Results:')
        prediction = model.predict(test_data)


        y_hat = np.argmax(prediction, axis=1)
        print('Accuracy: ', metrics.accuracy_score(test_labels, y_hat))
        print('Balanced Accuracy: ', metrics.balanced_accuracy_score(test_labels, y_hat))
        print('Macro F1', metrics.f1_score(test_labels, y_hat, average='macro'))
        print('Macro Precision', metrics.precision_score(test_labels, y_hat, average='macro'))
        print('Macro Recall', metrics.recall_score(test_labels, y_hat, average='macro'))

    if mode == 'training':
        print('Total time passed during training: ' + str(time.time() - startTime_s))

    # model.save(mahkeme + '_model.h5')