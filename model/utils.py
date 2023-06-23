import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
# Creat a padding mask where it returns 1.0 for non-zero inputs and 0.0 values for zero inputs
def create_padding_mask(seq):
    qmask=Lambda(lambda x:  K.cast(K.cast(x,'bool'),'float32'))(text_input)

# For prediction tasks:
# To ensure that each token can only attend to previous tokens and not future ones during training, you need to apply a look-ahead mask to the attention scores.
# The look-ahead mask is a binary matrix with zeros in the upper triangular part and ones in the lower triangular part.
# This mask is then multiplied element-wise with the attention scores to set the scores for future tokens to a large negative value (e.g., -inf).
# As a result, the softmax activation applied to the masked attention scores will effectively ignore future tokens.

def causal_attention_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask

def casual_attention_mask(seq_length):
    mask = np.triu(np.ones((seq_length, seq_length)), k=1)
    mask = np.where(mask == 1, -1e8, 0)
    return mask.astype(np.float32)