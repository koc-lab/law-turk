# law-turk
Predicting Decisions of Turkish Higher Courts.

Code to reproduce results given in "Natural Language Processing in Law: Prediction of outcomes in the Higher Courts of Turkey" by Emre Mumcuoğlu, Ceyhun E. Öztürk, Haldun M. Ozaktas and Aykut Koç (https://www.sciencedirect.com/science/article/abs/pii/S0306457321001692, https://doi.org/10.1016/j.ipm.2021.102684).

## Requirements
* scikit-learn
* tensorflow
* numpy
* pickle
* gensim

The deep learning models require the use of word embeddings. Download a Turkish word embedding model into _data_. You can use the one we used at https://github.com/akoksal/Turkish-Word2Vec

## Use

Simply call _predict.py_ with appropriate arguments.
* Court name: Should be one of _constitutional, civil, criminal, administrative, taxation, constitutional_right1, constitutional_right2, constitutional_right3, constitutional_right4, constitutional_right5, constitutional_right6, constitutional_right7_.
* Model name: Should be one of _Dummy, DT, RF, SVM, GRU, LSTM, BiLSTM_.
* Mode: Either _training_ or _test_. Use test mode to print test results after training.
* Optional argument --attention: Whether to use attention mechanism in deep learning models.

An example call:
> python3 predict.py constitutional BiLSTM training --attention
