# python3 lab4.py -v train.txt test.txt for Viterbi
# python3 lab4.py -b train.txt test.txt for Beam search

from collections import Counter
import argparse
import itertools
import numpy as np
import time, random
from sklearn.metrics import f1_score
import operator
import collections

parser = argparse.ArgumentParser()
parser.add_argument("-v", action = "store_true")
parser.add_argument("-b", action = "store_true")
parser.add_argument("trainData")
parser.add_argument("testData")
args = parser.parse_args()

random.seed(11242)
depochs = 5
feat_red = 0


### Load the dataset
def load_dataset_sents(file_path, as_zip=True, to_idx=False, token_vocab=None, target_vocab=None):
    targets = []
    inputs = []
    zip_inps = []
    with open(file_path) as f:
        for line in f:
            sent, tags = line.split('\t')
            words = [token_vocab[w.strip()] if to_idx else w.strip() for w in sent.split()]
            ner_tags = [target_vocab[w.strip()] if to_idx else w.strip() for w in tags.split()]
            inputs.append(words)
            targets.append(ner_tags)
            zip_inps.append(list(zip(words, ner_tags)))
    return zip_inps if as_zip else (inputs, targets)

train_data = load_dataset_sents(args.trainData)
test_data = load_dataset_sents(args.testData)

## unique tags
all_tags = ["O", "PER", "LOC", "ORG", "MISC"]

# feature space of cw_ct
def cw_ct_counts(data, freq_thresh=5):  # data inputted as (cur_word, cur_tag)

    cw_c1_c = Counter()

    for doc in data:
        cw_c1_c.update(Counter(doc))

    return Counter({k: v for k, v in cw_c1_c.items() if v > freq_thresh})

cw_ct_count = cw_ct_counts(train_data, freq_thresh=feat_red)


# # feature representation of a sentence cw-ct
def phi_1(sent, cw_ct_counts):  # sent as (cur_word, cur_tag)
    phi_1 = Counter()
    # include features only if found in feature space
    phi_1.update([item for item in sent if item in cw_ct_count.keys()])
    return phi_1

def pt_ct_counts(data, freq_thresh=5):  # input (cur_word, cur_tag)

    tagtag = Counter()

    for doc in data:

        tags = list(zip(*doc))[1]

        for i in range(len(tags)):

            if i == 0:

                tagtag.update([("*", tags[i])])

            else:

                tagtag.update([(tags[i - 1], tags[i])])

    # return feature space with features with counts above freq_thresh
    return Counter({k: v for k, v in tagtag.items() if v > freq_thresh})

pt_ct_count = pt_ct_counts(train_data, freq_thresh=feat_red)

class Perceptron():

    def __init__(self, all_tags):
        super(Perceptron, self).__init__()
        self.all_tags = all_tags

    # creating all possible combinaions of
    def pos_combos(self, sentence):
        # for p in itertools.product(self.all_tags, repeat=len(sentence)):
        #     print(p)
        combos = [list(zip(sentence, p)) for p in itertools.product(self.all_tags, repeat=len(sentence))]

        return combos

    def scoring(self, doc, weights, extra_feat=True):
        sentence, tags = list(zip(*doc))# unzippin them
        combos = list(enumerate(self.pos_combos(sentence)))# all possible combos of sequences [(wd,tag),(wd2,tag2)...]
        scores = np.zeros(len(combos))# our score matrix
        for index, sent_tag in combos:# looping through all possible combos
            phi = phi_1(sent_tag, cw_ct_count)# retrieving the counter if its in our feature space
            if len(phi) == 0:# if its not then the score is 0
                scores[index] = 0
            else:
                temp_score = 0
                for pair in phi:# otherwise do the w*local_phi
                    if pair in weights:
                        temp_score += weights[pair] * phi[pair]
                    else:
                        temp_score += 0

                scores[index] = temp_score# store the score with the index
        max_scoring_position = np.argmax(scores)# retrieve the index of the highest scoring sequence
        max_scoring_seq = combos[max_scoring_position][1]# retrieve the highest scoring sequence

        return max_scoring_seq

    def mViterbi(self, doc, weights, extra_feat=True):
        V = {}  # [[tags] ,[tags] ,[tags] ,[tags]]
        backptr = {}
        sentence, tags = list(zip(*doc))
        sentence_length = len(sentence)
        tag_length = len(self.all_tags)
        sentence += ("",)  # sentence[-1]
        V[0] = {}
        backptr[0] = {}
        for tag in range(len(self.all_tags)):#init the start word
            V[0][tag] = 0
            backptr[0][tag] = 0
        for n in range(1, sentence_length):  # 1 ~ N-1 don't need to do start word
            V[n] = {}
            backptr[n] = {}
            for candidate in range(tag_length):  # enum the tags of current word
                scores = {}
                for preTag in range(tag_length):# enum the tag of previous word
                    temp = [(sentence[n - 1], self.all_tags[preTag]),
                            (sentence[n], self.all_tags[candidate])]  # cw_cl , pw_pMaxL
                    mPhi = phi_1(temp, weights)
                    phiScore = 0
                    for pair in mPhi:#phi score
                        phiScore += mPhi[pair] + weights[pair]

                    scores[preTag] = V[n - 1][preTag] + phiScore

                maxTag = max(scores, key=lambda x: scores[x])#the max of (previous word,previous tag) + phi * weight
                V[n][candidate] = scores[maxTag]#update V
                backptr[n][candidate] = maxTag#update backptr
        answer = []
        lenn = len(V)
        temp = V[lenn - 1]
        a = max(temp, key=lambda x: temp[x])#trace back from the last word with the max score in V
        answer.append(self.all_tags[a])
        b = a
        for i in range(lenn - 1, 0, -1):#find back with backptr
            b = backptr[i][b]
            answer.append(self.all_tags[b])
        mm = []
        for i in reversed(answer):#reverse the answer
            mm.append(i)

        rr = []
        for i in range(len(mm)):#make prediction be feature of (word,tag)
            rr.append((sentence[i],mm[i]))
        return rr

    def mBeam(self, doc, weights, beamSize = 3,extra_feat=True):
        B = [(["start"], 0)]#init B
        sentence, tags = list(zip(*doc))
        sentence += ("",)
        for n in range(len(sentence) - 1):#sentence[-1]
            B2 = []
            for top in range(len(B)):  # top beam size of candidate
                temp = []
                for candidate in range(len(self.all_tags)):  # 5 tags
                    tagArr = B[top][0]  # [start,second wd, third wd.....]
                    toSearch = []
                    mPhi = {}
                    for i in range(len(B[top][0])):#count the sentence phi already in list
                        toSearch.append((sentence[i - 1], B[top][0][i]))
                    toSearch.append((sentence[len(B[top][0])-1],self.all_tags[candidate]))#the next word
                    mPhi = phi_1(toSearch, weights)
                    score = 0
                    for feature in mPhi:
                        score += weights[feature] + mPhi[feature]
                    score += B[top][1]

                    temp.append((self.all_tags[candidate], score))

                for iter in range(len(temp)):#update B2
                    temp2 = B[top][0] + [temp[iter][0]]

                    B2.append((temp2, temp[iter][1]))

            B = sorted(B2, key=lambda B2: B2[1], reverse=True)[:beamSize]

        B = sorted(B, key=lambda B2: B2[1],reverse=True)[:1]
        ans = B[0][0][1:]
        rr = []
        for i in range(len(ans)):
            rr.append((sentence[i], ans[i]))
        return rr

    def train_perceptron(self, data, epochs, shuffle=True, beamSize = 3, extra_feat=False):

        # variables used as metrics for performance and accuracy
        iterations = range(len(data) * epochs)
        false_prediction = 0
        false_predictions = []

        # initialising our weights dictionary as a counter
        # counter.update allows addition of relevant values for keys
        # a normal dictionary replaces the key-value pair
        weights = Counter()

        start = time.time()

        # multiple passes
        for epoch in range(epochs):
            false = 0
            now = time.time()

            # going through each sentence-tag_seq pair in training_data

            # shuffling if necessary
            if shuffle == True:
                random.shuffle(data)

            for doc in data:
                max_scoring_seq = None
                # retrieve the highest scoring sequence
                if args.v:
                    max_scoring_seq = self.mViterbi(doc, weights, extra_feat=extra_feat)
                if args.b:
                    max_scoring_seq = self.mBeam(doc, weights,beamSize = beamSize, extra_feat=extra_feat)
                # if the prediction is wrong
                if max_scoring_seq != doc:
                    correct = Counter(doc)

                    # negate the sign of predicted wrong
                    predicted = Counter({k: -v for k, v in Counter(max_scoring_seq).items()})

                    # add correct
                    weights.update(correct)

                    # negate false
                    weights.update(predicted)

                    """Recording false predictions"""
                    false += 1
                    false_prediction += 1
                false_predictions.append(false_prediction)

            print("Epoch: ", epoch + 1,
                  " / Time for epoch: ", round(time.time() - now, 2),
                  " / No. of false predictions: ", false)

        return weights, false_predictions, iterations

    # testing the learned weights
    def test_perceptron(self, data, weights, beamSize = 3,extra_feat=False):

        correct_tags = []
        predicted_tags = []

        i = 0

        for doc in data:
            _, tags = list(zip(*doc))

            correct_tags.extend(tags)

            max_scoring_seq = None
            # retrieve the highest scoring sequence
            if args.v:
                max_scoring_seq = self.mViterbi(doc, weights, extra_feat=extra_feat)
            if args.b:
                max_scoring_seq = self.mBeam(doc, weights, beamSize = 3, extra_feat=extra_feat)

            _, pred_tags = list(zip(*max_scoring_seq))

            predicted_tags.extend(pred_tags)

        return correct_tags, predicted_tags

    def evaluate(self, correct_tags, predicted_tags):

        f1 = f1_score(correct_tags, predicted_tags, average='micro', labels=["PER", "LOC", "ORG", "MISC"])

        print("F1 Score: ", round(f1, 5))

        return f1


perceptron = Perceptron(all_tags)
if args.v:
    print("Viterbi Algorithm : ")
    weights, false_predictions, iterations = perceptron.train_perceptron(train_data, epochs=depochs, extra_feat=True)

    correct_tags, predicted_tags = perceptron.test_perceptron(test_data, weights, extra_feat=True)

    f1 = perceptron.evaluate(correct_tags, predicted_tags)
if args.b:
    print("Beam Search Algorithm : ")
    for i in range(1,6):
        print("Beam size : ", i)
        weights, false_predictions, iterations = perceptron.train_perceptron(train_data, epochs=depochs,
                                                                             beamSize = i,extra_feat=True)
        correct_tags, predicted_tags = perceptron.test_perceptron(test_data, weights, beamSize = i,extra_feat=True)

        f1 = perceptron.evaluate(correct_tags, predicted_tags)


