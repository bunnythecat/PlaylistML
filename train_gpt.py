import sys
import os

cwd = os.getcwd()
sys.path.append(cwd)

import pickle
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from transformer import get_gpt, CosineDecay
import pandas as pd
import keras_nlp

def custom_loss(y_true, y_pred):
    mask = tf.math.not_equal(y_true, 0)  # Create a mask for non-zero values
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y_true, y_pred, sample_weight=mask)
    return loss

class MaskedSparseCategoricalAccuracy(tf.keras.metrics.SparseCategoricalAccuracy):
    def __init__(self, name='masked_sparse_categorical_accuracy', dtype=None):
        super(MaskedSparseCategoricalAccuracy, self).__init__(name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        mask = tf.math.not_equal(y_true, 0)  # Create a mask for non-zero values
        mask = tf.cast(mask, y_true.dtype)  # Cast the mask to the same dtype as y_true
        masked_y_true = tf.cast(y_true, y_true.dtype) * mask  # Apply the mask to y_true
        return super(MaskedSparseCategoricalAccuracy, self).update_state(masked_y_true, y_pred, sample_weight=sample_weight)


path = sys.argv[1]
model_save_path = path + "/ckpt"
# Load the data from the pickle file

with open((path + "/train.p"), "rb") as f:
    data = pickle.load(f)

with open((path + "/embedding_matrix.p"), "rb") as f:
    embedding_matrix = pickle.load(f)

with open((path + "/track_dictionary.p"), "rb") as f:
    track_dict = pickle.load(f)

# Define the maximum sentence length and input shape
DATA_SIZE = len(data)
MAX_SENT_LENGTH = 128
BATCH_SIZE = 48
VOCAB_SIZE = 115000
STEPS_PER_EPOCH = (DATA_SIZE - 1000) // BATCH_SIZE // 10
validation_steps = 1000 // BATCH_SIZE
min_lr = 1E-6

# Create a function to preprocess a playlist
def get_x_y(data):
    data = tf.concat([data, tf.zeros((MAX_SENT_LENGTH  + 1- tf.shape(data)[0],), dtype=tf.int32)], axis=0)
    return data[:-1], data[1:]

def vectorize(x, track_dict):
    return tf.py_function(lambda x: track_dict.get(x, 0), [x], tf.int32)

# Create a dataset from the data and preprocess it
dataset = tf.data.Dataset.from_generator(
    lambda: data,
    output_signature=tf.TensorSpec(shape=(None,), dtype=tf.int32),
)
dataset = dataset.repeat()
#dataset = dataset.map(lambda x: tf.reshape(x, shape=[MAX_SENT_LENGTH + 1]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset = dataset.map(get_x_y, num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Shuffle, batch, and prefetch the dataset
prefetch_size = tf.data.experimental.AUTOTUNE

data_size = len(data)
dataset = dataset.shuffle(buffer_size=BATCH_SIZE)

val = dataset.take(1000) 
train = dataset.skip(1000)

train = train.batch(BATCH_SIZE)
train = train.prefetch(prefetch_size)

val = train.batch(BATCH_SIZE)
val = train.prefetch(prefetch_size)


# Define the model
model = get_gpt(num_layers=8, d_model=256, num_heads=8, dff=256, input_vocab_size=VOCAB_SIZE,
                maximum_position_encoding=MAX_SENT_LENGTH, em_weights=embedding_matrix, rate=0.1, trainable=True)
model.summary()

warmup_steps = STEPS_PER_EPOCH * 2
learning_rate = CosineDecay(
    min_lr=min_lr, max_lr=min_lr * 6000, warmup_steps=warmup_steps
)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.999,
                                    epsilon=1e-9)

model.compile(
    optimizer, loss=custom_loss,
    metrics=[MaskedSparseCategoricalAccuracy(), 
    keras_nlp.metrics.Perplexity(
    from_logits=True, mask_token_id=0, name="perplexity")]
)

model_callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        model_save_path, save_best_only=False, verbose=True, save_weights_only=True),
]


history = model.fit(
    train,
    validation_data=val,
    verbose=1,
    epochs=20,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=validation_steps,
    callbacks=model_callbacks
)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()
hist.to_csv(path + "msp_gpt_log.csv")