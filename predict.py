

import argparse

import src.classify as classify
import src.deep as deep

parser = argparse.ArgumentParser()
parser.add_argument("court", help="Name of the court")
parser.add_argument("model_name", help="Model to be used")
parser.add_argument("mode", help='Mode: training or test')
parser.add_argument("--attention", action='store_true', help='Add if you want to use attention')
args = parser.parse_args()

print('Court name: ', args.court)
print('Model name: ', args.model_name)
print('Mode: ', args.mode)
print('Attention: ', args.attention)


court_names = ['constitutional', 'criminal', 'civil', 'administrative', 'taxation']
right_names = ['constitutional_right1', 'constitutional_right2', 'constitutional_right3', 'constitutional_right4', 'constitutional_right5', 'constitutional_right6', 'constitutional_right7']
model_names_classify = ['Dummy', 'DT', 'RF', 'SVM']
model_names_deep = ['GRU', 'LSTM', 'BiLSTM']
modes = ['training', 'test']


if args.court in court_names:
    if args.model_name in model_names_classify:
        classify.run_model(args.court, args.model_name, args.mode)
    elif args.model_name in model_names_deep:
        deep.run_model(args.court, args.model_name, args.mode, use_attention=args.attention)
    else:
        print('Invalid Model Name')
elif args.court in right_names:
    if args.model_name in model_names_classify:
        classify.run_model(args.court, args.model_name, args.mode)
    elif args.model_name in model_names_deep:
        print('You cannot use deep learning on right-based corpora')
    else:
        print('Invalid Model Name')
else:
    print('Invalid Court Name')