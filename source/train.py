___author__='Patrick Ruan'
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from nltk.tokenize import TweetTokenizer
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


tknzr = TweetTokenizer()

DATA_SIZE=10000
EMBEDDING_DIM=256
RNN_UNITS = 1024
BATCH_SIZE=20
SHUFFLE_BUFFER_SIZE=1000
TIME_STEPS=5
EPOCHS=20
THRED=200 # Thred of splitting dataset for training and testing, it is in range of (len(text)/time_steps/batch_size)
checkpoint_dir='./training_checkpoints'
saved_model_file='mymodel.h5'


class NextPredictor():

    def __init__(self,path='../phrases_data.txt',
                  embedding_dim=EMBEDDING_DIM,
                  rnn_units=RNN_UNITS,
                  batch_size=BATCH_SIZE,
                  shuffle_buffer_size=SHUFFLE_BUFFER_SIZE,
                  time_steps=TIME_STEPS,
                  epochs=EPOCHS,
                  checkpoint_dir=checkpoint_dir,
                  saved_model_file=saved_model_file,
                  thred=THRED):

        self._text=tknzr.tokenize(open(path, 'rb').read().decode(encoding='utf-8'))
        self.vocab=sorted(set(self._text))
        self._vocab_size=len(self.vocab)
        self._embedding_dim=embedding_dim
        self._rnn_units=rnn_units
        self._batch_size=batch_size
        self._shuffle_buffer_size=shuffle_buffer_size
        self._time_steps=time_steps
        self._checkpoint_dir=checkpoint_dir
        self._epochs=epochs
        self._saved_model_file=saved_model_file
        self._thred=thred

        self.word2idx={u:i for i, u in enumerate(self.vocab)}
        self.idx2word = np.array(self.vocab)

    def _split_input_target(self, sequence):
        # Creating input data and target data.
        input_text = sequence[:-1]
        target_text = sequence[1:]
        return input_text, target_text

    def _load_dataset(self):
        dataset_slices = tf.data.Dataset.from_tensor_slices(np.array([self.word2idx[c] for c in self._text]))
        sequences = dataset_slices.batch(self._time_steps+1, drop_remainder=True)
        dataset = sequences.map(self._split_input_target)
        new_dataset = dataset.shuffle(self._shuffle_buffer_size).batch(self._batch_size, drop_remainder=True)
        train_dataset=new_dataset.skip(self._thred)
        test_dataset = new_dataset.take(self._thred)
        return train_dataset,test_dataset

    def build_lstm(self,batch_size):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self._vocab_size, self._embedding_dim,
                                      batch_input_shape=[batch_size, None]),
            tf.keras.layers.Bidirectional(layers.LSTM(self._rnn_units, return_sequences=True)),
            tf.keras.layers.Dense(self._vocab_size)
        ])
        # print(model.summary())

        return model

    def _loss(self,labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

    def start_to_train(self):
        train_dataset, test_dataset=self._load_dataset()
        checkpoint_prefix = os.path.join(self._checkpoint_dir, "ckpt_{epoch}")
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,save_weights_only=True)
        model=self.build_lstm(self._batch_size)
        model.compile(optimizer='adam', loss=self._loss , metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

        model.fit(train_dataset, epochs=self._epochs, callbacks=[checkpoint_callback])
        model.save(self._saved_model_file)
        print ("Training Done! The model has been saved on "+self._saved_model_file)
        print ("Evaluating model on test dataset.......")
        model.evaluate(test_dataset)
        print("Done")

if __name__== "__main__":

    myPredictor = NextPredictor()
    myPredictor.start_to_train()