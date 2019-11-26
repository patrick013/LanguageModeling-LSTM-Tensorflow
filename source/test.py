___author__='Patrick Ruan'

import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from train import NextPredictor
from nltk.tokenize import TweetTokenizer
import numpy as np
import tensorflow as tf

tknzr = TweetTokenizer()
checkpoint_dir='./training_checkpoints'

predictor=NextPredictor()
model= predictor.build_lstm(batch_size=1)
latest=tf.train.latest_checkpoint(checkpoint_dir)
print("Using the latest trained weights from", latest)
model.load_weights(latest)
model.build(tf.TensorShape([1, None]))

def predict_next(start_string):
    for word in start_string.split():
        if word not in predictor.vocab:
            print(word," is not in our dataset!")
    input_eval = [predictor.word2idx[token] for token in tknzr.tokenize(start_string)]
    input_eval = tf.expand_dims(input_eval, 0)
    model.reset_states()
    predictions = model(input_eval)
    predictions = tf.squeeze(predictions, 0)
    pred_v = predictions[-1].numpy()

    # Finding the most potential next words
    i = 0
    candidates_index = []
    while i < 5:
        index = np.where(pred_v == np.max(pred_v))[0][0]
        candidates_index.append(index)
        pred_v[index] = np.min(pred_v)
        i += 1

    return np.array(candidates_index)

def _show_candidates(wordvector):
    strlist=[]
    index=1
    for w in wordvector:
        strlist.append(str(index)+": "+w)
        index+=1

    print('    '.join(strlist))

if __name__== "__main__":

    input_string = input("Type some thing and press enter (Case is sensitive): ")
    v = predict_next(input_string)
    word_vector=predictor.idx2word[v]
    _show_candidates(word_vector)
