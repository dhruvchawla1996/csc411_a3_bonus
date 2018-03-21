# Imports
import random
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


def build_sets():
    """
    Build training, testing and validation sets with labels 
    From content in clean_fake.txt and clean_real.txt

    PARAMETERS
    ----------
    None

    RETURNS
    -------
    training_set: list of list of strings
        contains headlines broken into words

    validation_set: list of list of strings
        contains headlines broken into words

    testing_set: list of list of strings
        contains headlines broken into words

    training_label: list of 0 or 1
        0 = fake news | 1 = real news for corresponding i-th element in training_set

    validation_label: list of 0 or 1
        0 = fake news | 1 = real news for corresponding i-th element in validation_set

    testing_label: list of 0 or 1
        0 = fake news | 1 = real news for corresponding i-th element in testing_set

    REQUIRES
    --------
    clean_fake.txt and clean_test.txt to be present in working directory
    """
    fname_fake, fname_real = "clean_fake.txt", "clean_real.txt"

    content_fake, content_real = [], []

    with open(fname_fake) as f:
        _content_fake = f.readlines()

        for line in _content_fake:
            content_fake.append(str.split(line))

    with open(fname_real) as f:
        _content_real = f.readlines()

        for line in _content_real:
            content_real.append(str.split(line))

    random.seed(42)

    random.shuffle(content_fake)
    random.shuffle(content_real)

    # Split in sets
    training_set, validation_set, testing_set = [], [], []
    training_label, validation_label, testing_label = [], [], []

    for i in range(len(content_fake)):
        if i < 0.7*len(content_fake):
            training_set.append(content_fake[i])
            training_label.append(0)
        elif i < 0.85*len(content_fake):
            validation_set.append(content_fake[i])
            validation_label.append(0)
        else:
            testing_set.append(content_fake[i])
            testing_label.append(0)

    for i in range(len(content_real)):
        if i < 0.7*len(content_real):
            training_set.append(content_real[i])
            training_label.append(1)
        elif i < 0.85*len(content_real):
            validation_set.append(content_real[i])
            validation_label.append(1)
        else:
            testing_set.append(content_real[i])
            testing_label.append(1)

    return training_set, validation_set, testing_set, training_label, validation_label, testing_label 

def word_to_index_builder(training_set, validation_set, testing_set):
    """
    Build a mapping of word -> unique number

    PARAMETERS
    ----------
    training_set: list of list of strings
        contains headlines broken into words

    validation_set: list of list of strings
        contains headlines broken into words

    testing_set: list of list of strings
        contains headlines broken into words

    RETURNS
    -------
    word_dict: {string, int}
        Matches word in training_set, validation_set, testing_set to a unique count number

    total_unique_words: int
        total number of unique words in training_set, validation_set, testing_set
    """
    word_dict = {}
    i = 0

    for headline in training_set+validation_set+testing_set:
        for word in headline:
            if word not in word_dict: 
                word_dict[word] = i
                i += 1

    return word_dict, i

def convert_sets_to_vector(training_set, validation_set, testing_set, training_label, validation_label, testing_label, word_index_dict, total_unique_words):
    """
    Convert training, validation and testing sets and labels from lists to numpy vectors 

    For each headline in a set:
    v[k] = 1 if kth word appears in the headline else 0

    For each label:
    [0, 1] if headline is fake else [1, 0]

    PARAMETERS
    ----------
    training_set, validation_set, testing_set: list of list of strings
        contains headlines broken into words

    training_label, validation_label, testing_label: list of 0 or 1
        0 = fake news | 1 = real news for corresponding i-th element in training_set

    word_index_dict: {string, int}
        Matches word in training_set, validation_set, testing_set to a unique count number

    total_unique_words: int
        total number of unique words in training_set, validation_set, testing_set

    RETURNS
    -------
    training_set_np, validation_set_np, testing_set_np: 
        numpy arrays representing conversions of training_set, validation_set, testing_set respectively

    training_label_np, validation_label_np, testing_label_np:
        numpy arrays representing conversions of training_label, validation_label, testing_label respectively
    """
    training_set_np, validation_set_np, testing_set_np = np.zeros((0, total_unique_words)), np.zeros((0, total_unique_words)), np.zeros((0, total_unique_words))

    # Training Set ############################################################
    for headline in training_set:
        training_set_i = np.zeros(total_unique_words)

        for word in headline:
            training_set_i[word_index_dict[word]] = 1.

        training_set_i = np.reshape(training_set_i, [1, total_unique_words])
        training_set_np = np.vstack((training_set_np, training_set_i))

    training_label_np = np.asarray(training_label).transpose()
    training_label_np_complement = 1 - training_label_np
    training_label_np = np.vstack((training_label_np, training_label_np_complement)).transpose()

    # Validation Set ############################################################
    for headline in validation_set:
        validation_set_i = np.zeros(total_unique_words)

        for word in headline:
            validation_set_i[word_index_dict[word]] = 1.

        validation_set_i = np.reshape(validation_set_i, [1, total_unique_words])
        validation_set_np = np.vstack((validation_set_np, validation_set_i))

    validation_label_np = np.asarray(validation_label).transpose()
    validation_label_np_complement = 1 - validation_label_np
    validation_label_np = np.vstack((validation_label_np, validation_label_np_complement)).transpose()

    # Testing Set ############################################################
    for headline in testing_set:
        testing_set_i = np.zeros(total_unique_words)

        for word in headline:
            testing_set_i[word_index_dict[word]] = 1.

        testing_set_i = np.reshape(testing_set_i, [1, total_unique_words])
        testing_set_np = np.vstack((testing_set_np, testing_set_i))

    testing_label_np = np.asarray(testing_label).transpose()
    testing_label_np_complement = 1 - testing_label_np
    testing_label_np = np.vstack((testing_label_np, testing_label_np_complement)).transpose()

    return training_set_np, validation_set_np, testing_set_np, training_label_np, validation_label_np, testing_label_np

def ready_the_data():
    training_set, validation_set, testing_set, training_label, validation_label, testing_label  = build_sets()
    word_index_dict, total_unique_words = word_to_index_builder(training_set, validation_set, testing_set)
    training_set_np, validation_set_np, testing_set_np, training_label_np, validation_label_np, testing_label_np = convert_sets_to_vector(training_set, validation_set, testing_set, training_label, validation_label, testing_label, word_index_dict, total_unique_words)

    return training_set_np, validation_set_np, testing_set_np, training_label, validation_label, testing_label

################################################################################
################################################################################
################################################################################

def build_set_tfidf():
    fname_fake, fname_real = "clean_fake.txt", "clean_real.txt"

    content_fake, content_real = [], []

    with open(fname_fake) as f:
        _content_fake = f.readlines()

        for line in _content_fake:
            content_fake.append(line[:-1])

    with open(fname_real) as f:
        _content_real = f.readlines()

        for line in _content_real:
            content_real.append(line[:-1])

    random.seed(42)

    random.shuffle(content_fake)
    random.shuffle(content_real)

    # Split in sets
    training_set, validation_set, testing_set = [], [], []
    training_label, validation_label, testing_label = [], [], []

    for i in range(len(content_fake)):
        if i < 0.7*len(content_fake):
            training_set.append(content_fake[i])
            training_label.append(0)
        elif i < 0.85*len(content_fake):
            validation_set.append(content_fake[i])
            validation_label.append(0)
        else:
            testing_set.append(content_fake[i])
            testing_label.append(0)

    for i in range(len(content_real)):
        if i < 0.7*len(content_real):
            training_set.append(content_real[i])
            training_label.append(1)
        elif i < 0.85*len(content_real):
            validation_set.append(content_real[i])
            validation_label.append(1)
        else:
            testing_set.append(content_real[i])
            testing_label.append(1)

    tfidf_vectorizer = TfidfVectorizer(stop_words = "english", max_df = 0.95)

    training_set = tfidf_vectorizer.fit_transform(training_set)
    validation_set = tfidf_vectorizer.transform(validation_set)
    testing_set = tfidf_vectorizer.transform(testing_set)

    return training_set, validation_set, testing_set, training_label, validation_label, testing_label

def build_set_count():
    fname_fake, fname_real = "clean_fake.txt", "clean_real.txt"

    content_fake, content_real = [], []

    with open(fname_fake) as f:
        _content_fake = f.readlines()

        for line in _content_fake:
            content_fake.append(line[:-1])

    with open(fname_real) as f:
        _content_real = f.readlines()

        for line in _content_real:
            content_real.append(line[:-1])

    random.seed(42)

    random.shuffle(content_fake)
    random.shuffle(content_real)

    # Split in sets
    training_set, validation_set, testing_set = [], [], []
    training_label, validation_label, testing_label = [], [], []

    for i in range(len(content_fake)):
        if i < 0.7*len(content_fake):
            training_set.append(content_fake[i])
            training_label.append(0)
        elif i < 0.85*len(content_fake):
            validation_set.append(content_fake[i])
            validation_label.append(0)
        else:
            testing_set.append(content_fake[i])
            testing_label.append(0)

    for i in range(len(content_real)):
        if i < 0.7*len(content_real):
            training_set.append(content_real[i])
            training_label.append(1)
        elif i < 0.85*len(content_real):
            validation_set.append(content_real[i])
            validation_label.append(1)
        else:
            testing_set.append(content_real[i])
            testing_label.append(1)

    count_vectorizer = CountVectorizer(stop_words = "english", max_df = 0.95)

    training_set = count_vectorizer.fit_transform(training_set)
    validation_set = count_vectorizer.transform(validation_set)
    testing_set = count_vectorizer.transform(testing_set)

    return training_set, validation_set, testing_set, training_label, validation_label, testing_label