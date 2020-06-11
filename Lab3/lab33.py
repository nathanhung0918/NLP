# !/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Weishi Li
# time: 2019-03-13

import sys
from collections import Counter
from sklearn.metrics import f1_score
from itertools import chain


# -------------------Function--------------------
# Bigrams sequence
# -----------------------------------------------
def bigrams(seq):
    seq_bigram = [b for b in zip(seq[:-1], seq[1:])]
    return seq_bigram

# ----------------------------------------------
#        Function: Load the Data
# ----------------------------------------------
def load_dataset_sents(file_path, as_zip=True, to_idx=False, token_vocab=None, target_vocab=None):
    targets = []
    inputs = []
    zip_inps = []
    with open(file_path) as f:
        for line in f:
            sent, tags = line.split('\t')
            words = [token_vocab[w.strip()] if to_idx else w.strip() for w in sent.split()]
            ner_tags = [target_vocab[w.strip()] if to_idx else w.strip() for w in tags.split()]
            inputs.append(words)  # list of words
            targets.append(ner_tags)  # list of tags
            zip_inps.append(list(zip(words, ner_tags)))  # list of (words,tags)
    return zip_inps if as_zip else (inputs, targets)

# ----------------------------------------------
#        Feature Extraction
#        Input : corpus c = [s1, s2, ... , sn]
#        type: phi_1, phi_2
# ----------------------------------------------
# ----------------------------------------------
#        Function: cw_cl_extract
#        Feature: current word - current lable
# ----------------------------------------------
def cw_cl_extract(train_data):
    cw_cl_counts = {}
    for sent in train_data:
        for word_tag in sent:
            if word_tag not in cw_cl_counts:
                cw_cl_counts[word_tag] = 0
            cw_cl_counts[word_tag] += 1
    for k in list(cw_cl_counts):
        if cw_cl_counts[k] <= 2:
            del cw_cl_counts[k]
    return cw_cl_counts

# ----------------------------------------------
#        Function: phi_1
#        word_tag_features of the sentence
# ----------------------------------------------
def phi_1(words, tags, cw_cl_counts):
    sent = list(zip(words, tags))
    sent_counts = Counter(sent)
    for cw_cl in sent:
        if cw_cl not in cw_cl_counts:
            del sent_counts[cw_cl]
    return sent_counts


# ----------------------------------------------
#        Function: pl_cl_extract
#        Feature: previous label - current label
# ----------------------------------------------
def pl_cl_extract(train_data):
    pl_cl_counts = {}
    for sent in train_data:
        tag_sent = [i[1] for i in sent]
        tag_sent = ["None"] + tag_sent
        tag_sent = bigrams(tag_sent)
        for pl_cl in tag_sent:
            if pl_cl not in pl_cl_counts:
                pl_cl_counts[pl_cl] = 0
            pl_cl_counts[pl_cl] += 1

    for k in list(pl_cl_counts):
        if pl_cl_counts[k] <= 2:
            del pl_cl_counts[k]
    return pl_cl_counts


# ----------------------------------------------
#        Function: phi_2
#        tag_tag_features of the sentence
# ----------------------------------------------
def phi_2(tags, pl_cl_counts):
    pl_cl_sent = bigrams(["None"] + tags)
    sent_counts = Counter(pl_cl_sent)
    for pl_cl in pl_cl_sent:
        if pl_cl not in pl_cl_counts:
            del sent_counts[pl_cl]
    return sent_counts


# ----------------------------------------------
#        Function: phi
#        phi_1 + phi_2
# ----------------------------------------------
def phi(words, tags, cw_cl_counts, pl_cl_counts, feature):
    feature1 = phi_1(words, tags, cw_cl_counts)

    if feature == 2:
        phi2 = phi_2(tags, pl_cl_counts)
        feature1.update(phi2)
        feature2 = feature1
        return feature2
    else:
        return feature1


# ----------------------------------------------
#        Function: Structured_perceptron
# ----------------------------------------------

def structured_perceptron(train_data, feature):
    W = {}
    if feature == "cw_cl":
        W = train(train_data, 1)

    if feature == "pl_cl":
        W = train(train_data, 2)
    return W


# ----------------------------------------------
#        Function: train
# ----------------------------------------------
def train(train_data, feature):
    W = {}
    # obtain max_length of sentence in train_data
    max_len = 0
    for sent in train_data:
        length = len(sent)
        if length > max_len:
            max_len = length

    # obtain a dictionary of tables_tags of different length(key)
    Table_tags = make_tags(max_len)

    for sent in train_data:

        words = [i[0] for i in sent]
        tags = [i[1] for i in sent]
        table_tag = Table_tags[len(sent) - 1]

        # compare y and y_pre
        y_true = tags
        y_pre = predict(words, table_tag, W, feature)
        #
        # print()
        # print(y_true)
        # print(y_pre)
        # print(W)
        # print(y_true,":::::::::",y_pre)
        if y_pre != y_true:
            feat_correct = phi(words, y_true, cw_cl_counts, pl_cl_counts, feature)
            feat_predict = phi(words, y_pre, cw_cl_counts, pl_cl_counts, feature)
            feat_diff = Counter(feat_correct)
            feat_diff.subtract(feat_predict)
            # print("feat_correct: ", feat_correct)
            # print("feat_predict: ", feat_predict)
            # print("feat_diff: ", feat_diff)
            for i in feat_diff:
                if i not in W:
                    W[i] = 0
                W[i] += feat_diff[i]
    return W


# ----------------------------------------------
#        Function: predict
# ----------------------------------------------
def predict(words, table_tag, W, feature):

    tags_predict = table_tag[0]
    max_score = 0

    for tags in table_tag:
        Feature = phi(words, tags, cw_cl_counts, pl_cl_counts, feature)
        score = 0

        for key in Feature:
            if key in W:
                score += W[key] * Feature[key]
        if (score > max_score):
            max_score = score
            tags_predict = tags

    return tags_predict


# ----------------------------------------------
#        Function: make_tags
# ----------------------------------------------
def make_tags(max_len):
    TAGS = ["O", "PER", "LOC", "ORG", "MISC"]
    table_y = [["O"], ["PER"], ["LOC"], ["ORG"], ["MISC"]]
    Table_tag = {}
    for i in range(max_len):
        new = []
        for j in table_y:
            for k in TAGS:
                new.append(j + [k])

        Table_tag[i] = table_y
        table_y = new
    return Table_tag


# ----------------------------------------------
#        Function: test
# ----------------------------------------------
def test(W, words, tags):
    correct = tags
    # flatten
    correct = list(chain.from_iterable(correct))

    max_len = 0
    for sent in test_data:
        length = len(sent)
        if length > max_len:
            max_len = length
    # obtain a dictionary of tables_tags of different length(key)
    Table_tags = make_tags(max_len)

    y_pre = []
    feature = 1
    for words in words_test:
        n = len(words)
        table_tag = Table_tags[n - 1]
        y_pre.append(predict(words, table_tag, W, feature))
    pre = y_pre

    # flatten
    predicted = list(chain.from_iterable(pre))

    # evaluation : micro-averaged F1 score
    f1_micro = f1_score(correct, predicted, average='micro', labels=['ORG', 'MISC', 'PER', 'LOC'])
    return f1_micro, predicted, correct

# ----------------------------------------------
#        Function: Main
# ----------------------------------------------
train_file = sys.argv[1]
test_file = sys.argv[2]

train_data = load_dataset_sents(train_file)
cw_cl_counts = cw_cl_extract(train_data)
pl_cl_counts = pl_cl_extract(train_data)

#------------------------------------------------
#      Feature: current word _ current label
#------------------------------------------------
# train model
W = structured_perceptron(train_data, feature="pl_cl")


# make prediction use "test.txt"
test_data = load_dataset_sents(test_file, as_zip=True)
words_test, tags_test = load_dataset_sents(test_file, as_zip=False)

# evaluate
f1_micro, pre, cor = test(W, test_data, tags_test)
print("Feature:   Current word _ current label\nF1 Score: ", f1_micro)




