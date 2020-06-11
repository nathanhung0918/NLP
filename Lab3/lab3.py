# python3 lab3.py train.txt test.txt

import sys,time,operator
from random import shuffle
from sklearn.metrics import f1_score
class NER:
    def __init__(self):
        self.train_file = sys.argv[1]  # training data
        self.test_file = sys.argv[2]  # testing data

    def load_dataset_sents(self,file_path, as_zip=True, to_idx=False, token_vocab=None, target_vocab=None):
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

    def current_word_current_label(self,corpus):
        cw_cl_counts = {}
        temp = {}
        for sentence in corpus:
            for wd,tag in sentence:
                combine = wd + "_" + tag #cw_cl_feature
                if combine not in temp:#count
                    temp[combine] = 1
                else:
                    temp[combine] += 1
        for keys in temp:#delete value < 3
            if temp[keys] >= 3 :
                cw_cl_counts[keys] = temp[keys]
        return cw_cl_counts

    def phi_1(self,x, y, cw_cl_counts):
        phi = {}
        for i in range(len(y)):#?????? x ?
            combine = str(x[i]) + "_" + str(y[i])#cw_cl_feature
            if combine in cw_cl_counts:
                if combine in phi:
                    phi[combine] += 1
                else:
                    phi[combine] = 1
        return phi

    def previous_label_current_label(self,corpus):
        temp = {}
        pl_cl_counts = {}
        pre = ""#previous tag
        for i in corpus:
            for wd, tag in i:
                combine = pre + "_" + tag
                pre = tag
                if combine not in temp:#count
                    temp[combine] = 1
                else:
                    temp[combine] += 1
        for keys in temp:#delete < 3
            if temp[keys] >= 3 :
                pl_cl_counts[keys] = temp[keys]
        return pl_cl_counts

    def phi_2(self,x, y, pl_cl_counts):
        phi = {}
        pre = ""#previous label
        for i in range(len(y)):
            combine = pre + "_" + y[i]
            pre = y[i]
            if combine in pl_cl_counts:
                if combine in phi:
                    phi[combine] += 1
                else:
                    phi[combine] = 1
        return phi

    def train1(self,weight):
        x,y = self.load_dataset_sents(self.train_file,False) #training data
        for i in range(len(x)):
            yActual = y[i]
            yPredict = self.predict(x[i],y[i],weight)

            # print(yActual,"::::",yPredict)

            if yActual != yPredict:#update
                predictList = self.phi_1(x[i],yPredict,weight)
                actualList = self.phi_1(x[i],yActual,weight)

                for keys in actualList:
                    weight[keys] += actualList[keys]
                for keys in predictList:
                    weight[keys] -= predictList[keys]
        return weight

    def predict(self,x,y,weight): # sentence = dict / weight = dict
        mPredict = []#final predict
        candidate = ['O','PER','LOC','ORG','MISC']#all candidate
        random_sequence = []# 4word: 0000-4444
        maxSize = 1 #625
        maxValue = -1
        maxSequence = ""#store the max possibility
        for i in range(len(x)):
            maxSize *= len(candidate)
        for i in range(maxSize):
            random_sequence.append(self.decto5(i,5))
        shuffle(random_sequence)#shuffle
        for num in random_sequence:#625 possibility
            toSearch = []
            total = 0
            for i in num:#string ex: 4401
                toSearch.append(candidate[int(i)])#correct
            result = self.phi_1(x,toSearch,weight)#phi_feature

            for keys in result:#count score
                total += result[keys] * weight[keys]

            if total > maxValue:#find argmax
                maxSequence = num
                maxValue = total
        # print(maxSequence)
        temp = ""
        if len(maxSequence) < len(x):
            for i in range(len(x)-len(maxSequence)):
                temp += '0'
            maxSequence = temp + maxSequence
        for i in maxSequence:#store prediction
            mPredict.append(candidate[int(i)])

        return mPredict

    def decto5(self, n, x): #dec to 5
        a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        b = []
        mString = ""
        while True:
            s = n // x  # Quantity
            y = n % x  # Rest
            b = b + [y]
            if s == 0:
                break
            n = s
        b.reverse()
        for i in b:
            mString += str(a[i])
        return mString

    def train2(self,weight):
        x,y = self.load_dataset_sents(self.train_file,False) #training data
        for i in range(len(x)):
            yActual = y[i]
            yPredict = self.predict(x[i],y[i],weight)

            # print(yActual,"::::",yPredict)

            if yActual != yPredict:#update
                predictList = self.phi_2(x[i],yPredict,weight)
                actualList = self.phi_2(x[i],yActual,weight)

                for keys in actualList:
                    weight[keys] += actualList[keys]
                for keys in predictList:
                    weight[keys] -= predictList[keys]
        return weight

    def predict2(self,x,y,weight): # sentence = dict / weight = dict
        mPredict = []#final predict
        candidate = ['O','PER','LOC','ORG','MISC']#all candidate
        random_sequence = []# 4word: 0000-4444
        maxSize = 1 #625
        maxValue = -1
        maxSequence = ""#store the max possibility
        for i in range(len(x)):
            maxSize *= len(candidate)
        for i in range(maxSize):
            random_sequence.append(self.decto5(i,5))
        shuffle(random_sequence)#shuffle
        for num in random_sequence:#625 possibility
            toSearch = []
            total = 0
            for i in num:#string ex: 4401
                toSearch.append(candidate[int(i)])#correct
            result = self.phi_2(x,toSearch,weight)#phi_feature

            for keys in result:#count score
                total += result[keys] * weight[keys]

            if total > maxValue:#find argmax
                maxSequence = num
                maxValue = total
        # print(maxSequence)
        temp = ""
        if len(maxSequence) < len(x):
            for i in range(len(x)-len(maxSequence)):
                temp += '0'
            maxSequence = temp + maxSequence
        for i in maxSequence:#store prediction
            mPredict.append(candidate[int(i)])

        return mPredict

    def eval(self):
        weight = self.previous_label_current_label(self.load_dataset_sents(self.train_file))#original dictionary
        for keys in weight:  # set weight to 0
            weight[keys] = 0
        for i in range(1):#multi pass
            weight = self.train2(weight)
        x,y = self.load_dataset_sents(self.test_file,False) #testing data
        correct = []
        predicted = []
        for i in range(len(y)):
            mPredict = self.predict(x[i],y[i],weight)
            for j in mPredict:
                predicted.append(j)
            for j in y[i]:
                correct.append(j)
        print("10 Top positive",sorted(weight.items(), key=operator.itemgetter(1),reverse=True)[:10])
        f1_micro = f1_score(correct, predicted, average='micro', labels=['ORG', 'MISC', 'PER', 'LOC'])
        print("pl_cl: ",f1_micro)

        weight = self.current_word_current_label(self.load_dataset_sents(self.train_file))  # original dictionary
        for keys in weight:  # set weight to 0
            weight[keys] = 0
        for i in range(1):  # multi pass
            weight = self.train1(weight)
        x, y = self.load_dataset_sents(self.test_file, False)  # testing data
        correct = []
        predicted = []
        for i in range(len(y)):
            mPredict = self.predict(x[i], y[i], weight)
            for j in mPredict:
                predicted.append(j)
            for j in y[i]:
                correct.append(j)
        print("10 Top positive", sorted(weight.items(), key=operator.itemgetter(1), reverse=True)[:10])
        f1_micro = f1_score(correct, predicted, average='micro', labels=['ORG', 'MISC', 'PER', 'LOC'])
        print("cw_cl: ", f1_micro)


if __name__ == '__main__':
    ner = NER()
    ner.eval()




