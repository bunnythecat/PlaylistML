"""This module defines a BentoML service that uses a Keras model to classify
digits.
"""

import numpy as np
import bentoml
from bentoml.io import NumpyNdarray
import tensorflow as tf

def sample_from(logits):
    logits, indices = tf.math.top_k(logits, k=10, sorted=True)
    indices = np.asarray(indices).astype("int32")
    preds = tf.keras.activations.softmax(tf.expand_dims(logits, 0))[0]
    preds = np.asarray(preds).astype("float32")
    return np.random.choice(indices, p=preds)

BENTO_MODEL_TAG = "keras_lmrec:jtxvnoaqh2a33sd7"
model = bentoml.keras.get(BENTO_MODEL_TAG)
lmrec_runner = model.to_runner()
lmrec_service = bentoml.Service("keras_lmrec", runners=[lmrec_runner])

@lmrec_service.api(
    input=NumpyNdarray(
        shape=(-1, 128),
        dtype=np.int32,
        enforce_dtype=True,
        enforce_shape=True
    ),
    output=NumpyNdarray())
def lmrec_predict(input_data: np.ndarray) -> np.ndarray:
    indices = np.where(input_data[0] == 0)[0][0]
    return sample_from(lmrec_runner.predict.run(input_data)[0][indices - 1])