import json

import numpy as np
import requests

import sys
import pickle
import tensorflow as tf

class TokenGenerator():
    """Generate text from a trained autoregressive model.
    1. Feed some starting prompt to the model
    2. Predict probabilities for the next token
    3. Sample the next token and add it to the next input

    Arguments:
        max_tokens: Integer, the number of tokens to be generated after prompt.
        start_tokens: List of integers, the token indices for the starting prompt.
        index_to_word: List of strings, obtained from the TextVectorization layer.
        top_k: Integer, sample from the `top_k` token predictions.
    """

    def __init__(self, max_len, track_dict, SERVICE_URL, top_k=20):
        self.max_len = max_len
        self.track_dict = track_dict
        self.index_to_word = list(self.track_dict)
        self.SERVICE_URL = SERVICE_URL
        self.k = top_k

    def tokenize(self, tokens):
        return [track_dict.get(_, 0) for _ in tokens]

    def detokenize(self, number):
        return self.index_to_word[number]

    def make_request_to_bento_service(
        self, input_array: np.ndarray
    ) -> str:
        serialized_input_data = json.dumps(input_array.tolist())
        response = requests.post(
            self.SERVICE_URL,
            headers={"content-type": "application/json"},
            data=serialized_input_data,
        )
        return response.text

    def generate(self, start_tokens, n, logs=None):
        start_tokens = [self.track_dict.get(_,0) for _ in start_tokens]
        if len(start_tokens) > 0:
            tokens_generated = []
            while len(tokens_generated) < n:
                pad_len = self.max_len - len(start_tokens)
                sample_index = len(start_tokens) - 1
                if pad_len < 0:
                    x = start_tokens[:self.max_len]
                    sample_index = self.max_len - 1
                elif pad_len > 0:
                    x = start_tokens + [0] * pad_len
                else:
                    x = start_tokens
                x = np.array([x])
                y = self.make_request_to_bento_service(x)
                y = np.array([y])
                y = int(y[0])            
                if y not in tokens_generated:
                    tokens_generated.append(y)
                    start_tokens.append(y)
            txt = ",".join(
                [self.detokenize(_) for _ in tokens_generated]
            )
            return txt

def make_request_to_bento_service(
    service_url: str, input_array: np.ndarray
) -> str:
    serialized_input_data = json.dumps(input_array.tolist())
    response = requests.post(
        service_url,
        data=serialized_input_data,
        headers={"content-type": "application/json"}
    )
    return response.text

if __name__ == "__main__":
    SERVICE_URL = "http://localhost:3000/lmrec_predict"
    # Define the maximum sentence
    MAX_SENT_LENGTH = 128

    # Load track dictionary
    path = sys.argv[1]
    with open((path + "/track_dictionary.p"), "rb") as f:
        track_dict = pickle.load(f)


    start_prompt = "0Aqi7ArnBrGblW5T6p2jmD,6J17MkMmuzBiIOjRH6MOBZ,0bVtevEgtDIeRjCJbK3Lmv,7EZC6E7UjZe63f1jRmkWxt,6xGruZOHLs39ZbVccQTuPZ"
    start_tokens = start_prompt.split(",")

    print(start_tokens)

    num_tokens_generated = 5
    tokenGenerator = TokenGenerator(MAX_SENT_LENGTH, track_dict, SERVICE_URL)
    text = tokenGenerator.generate(start_tokens, num_tokens_generated)
    print("generated Tokens:\n{}\n".format(text))