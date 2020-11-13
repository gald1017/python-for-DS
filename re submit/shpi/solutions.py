import numpy as np
import torch
from sklearn.metrics import mean_squared_error;
import lorem
import re
import string

def notify_result(problem_id, correct):
    if (correct):
        print('Problem ' + str(problem_id) + ': Correct')
    else:
        print('Problem ' + str(problem_id) + ': Incorrect')

def generate_data(start, end, size):
    shape = np.random.randint(start, end, size=size)
    return np.random.normal(size=shape[0] * shape[1]).reshape(shape)

def recm_naive(ts, eps):
    ln = len(ts)

    rm = np.zeros((ln, ln), dtype=bool)
    for i in range(ln):
        for j in range(ln):
            rm[i, j] = np.abs(ts[i]-ts[j])<eps
    return rm

def junk_to_ascii(s):
  str = re.sub(r"\s*[^A-Za-z]+\s*", " ", s)
  return str.strip().lower()

def encode_word_naive(str, alphabet=string.ascii_lowercase):
    vector = [[0 if char != letter else 1 for char in alphabet] 
                  for letter in str]
    return torch.tensor(vector).float()

def encode_naive(words):
    n = len(words)
    max_l = max(map(len, words))
    encodings = torch.zeros((n, max_l, 26))
    for i, word in enumerate(words):
        encoding = encode_word_naive(word)
        encodings[i][torch.arange(len(word))] = encoding
    return encodings

def check(student, problem_id, func_to_check):
    correct = False
    
    if (problem_id == 1):
        for i in range(3):
            data = generate_data(10, 100, 2)
            actual = func_to_check(data)
            for i, row in enumerate(actual):
                expected_row = (data[i] - np.mean(data[i])) / np.std(data[i])
                correct = np.allclose(expected_row, row)
                if (not correct):
                    break;
            if (not correct):
                    break;
    elif (problem_id == 2):
        for i in range(3):
            data = generate_data(10, 100, 2)
            actual = func_to_check(data)
            for i, column in enumerate(actual.T):
                expected_column = (data[:, i] - np.mean(data[:, i])) / np.std(data[:, i])
                correct = np.allclose(expected_column, column)
                if (not correct):
                    break;
            if (not correct):
                    break;
    elif (problem_id == 3):
        for i in range(3):
            data = generate_data(10, 100, 2)
            normalized_columns = func_to_check(data, 0)
            normalized_rows = func_to_check(data, 1)
            for i, row in enumerate(normalized_rows):
                expected_row = (data[i] - np.mean(data[i])) / np.std(data[i])
                correct = np.allclose(expected_row, row)
                if (not correct):
                    break;
            for i, column in enumerate(normalized_columns.T):
                expected_column = (data[:, i] - np.mean(data[:, i])) / np.std(data[:, i])
                correct = np.allclose(expected_column, column)
                if (not correct):
                    break;
            if (not correct):
                    break;
    elif (problem_id == 4):
        for i in range(3):
            m = generate_data(10, 100, 2)
            v = np.random.randint(10, 100, size=m.shape[1])
            expected = np.dot(m, v)
            actual = func_to_check(m, v)
            correct = np.allclose(actual, expected)
            if (not correct):
                break;
    elif (problem_id == 5):
        for i in range(3):
            size = np.random.randint(100, 1000)
            ts = np.random.normal(size=size)
            eps = np.random.normal()
            expected = recm_naive(ts, eps)
            actual = func_to_check(ts, eps)
            correct = np.array_equal(actual, expected)
            if (not correct):
                break;
    elif (problem_id == 6):
        relu = torch.nn.ReLU()
        for i in range(3):
            n, m, k = np.random.randint(10, 100, size=3)
            input = torch.randn(size=(n, m, k))
            expected = relu(input)
            actual = func_to_check(input)
            correct = torch.equal(actual, expected)
            if (not correct):
                break;
    elif (problem_id == 7):
        for i in range(10):
            n = np.random.randint(100, 1000)
            s = np.random.randint(0, 3)
            y_pred = None
            if (s == 0):
                y_pred = torch.from_numpy(np.random.normal(size=(n, 1)))
            elif (s == 1):
                y_pred = torch.from_numpy(np.random.normal(size=(1, n)))
            else:
                y_pred = torch.from_numpy(np.random.normal(size=n))
            successes = np.random.randint(n)
            indices = np.random.randint(n, size=successes)
            y_true = torch.zeros(n)
            y_true[indices] = 1
            expected = mean_squared_error(y_true, y_pred.view(n))
            actual = func_to_check(y_true, y_pred)
            correct = np.allclose(expected, actual)
            if (not correct):
                break;
    elif (problem_id == 8):
        for i in range(3):
            text = junk_to_ascii(lorem.text())
            words = text.split()
            expected = encode_naive(words)
            actual = func_to_check(words)
            correct = torch.equal(actual, expected)
            if (not correct):
                break;
    notify_result(problem_id, correct)
    return 0
