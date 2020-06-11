# python3 lab1.py review_polarity
import sys, re, os, random, operator,time
from itertools import islice
import matplotlib.pyplot as plt
import numpy as np


class Perceptron:
    def __init__(self):
        self.folder = sys.argv[1]  # folder name
        self.neg = os.listdir(self.folder + "/txt_sentoken" + "/neg")  # neg and pos candidate
        self.neg = sorted(self.neg)  # sort the file name in order
        self.pos = os.listdir(self.folder + "/txt_sentoken" + "/pos")
        self.pos = sorted(self.pos)
        self.counts = {}  # tf of single document
        self.weight = {}  # total weight
        self.score = 0  # train score
        self.train_n = {}  # training data (negative)
        self.train_p = {}
        self.fit_n = {}
        self.fit_p = {}  # fitting data (positive)
        for name in self.neg[0:800]:  # 800 for training
            self.train_n[self.folder + "/txt_sentoken" + "/neg" + "/" + name] = -1
        for name in self.pos[0:800]:  # 800 for training
            self.train_p[self.folder + "/txt_sentoken" + "/pos" + "/" + name] = 1
        for name in self.neg[800:1000]:  # 200 fitting negative
            self.fit_n[self.folder + "/txt_sentoken" + "/neg" + "/" + name] = -1
        for name in self.pos[800:1000]:  # 200 fitting positive
            self.fit_p[self.folder + "/txt_sentoken" + "/pos" + "/" + name] = 1
        self.container = []  # all data container
        self.train_data = {}  # {(tf: wd:freq),sentiment}
        self.valid_data = {}
        self.correct_validate = 0  # total correct prediction
        self.y_hat = 0  # init prediction
        self.c = 1
        self.weight_c = {}  # the weight of each train
        self.result = []

    def bagOfWords(self, filePath, sentiment):  # make weight dict, 
        wordRE = re.compile(r'[A-Za-z]+')  # tokenizer
        self.counts = {}  # empty container
        with open(filePath, 'r') as infile:  # read file
            for line in infile:
                for word in wordRE.findall(line):
                    if word not in self.counts:  # not in,set to 1
                        self.counts[word] = 1
                    else:
                        self.counts[word] += 1  # is in, accumulate
                    if word not in self.weight:  # init weight to 0
                        self.weight[word] = 0
        self.container.append((self.counts, sentiment))

    # def bigram(self, filePath, sentiment):  # make weight dict #run for too long
    #     wordRE = re.compile(r'[A-Za-z]+')  # tokenizer
    #     self.counts = {}  # empty container
    #     with open(filePath, 'r') as infile:  # read file
    #         for line in infile:
    #             words = wordRE.findall(line)
    #             a = zip(words, islice(words, 1, None))
    #             for word in a:
    #                 if word not in self.counts:  # not in,set to 1
    #                     self.counts[word] = 1
    #                 else:
    #                     self.counts[word] += 1  # is in, accumulate
    #                 if word not in self.weight:  # init weight to 0
    #                     self.weight[word] = 0.0
    #     self.container.append((self.counts, sentiment))

    def bigram(self, filePath, sentiment):  # make weight, dict enhanced by combine train and average
        wordRE = re.compile(r'[A-Za-z]+')  # tokenizer
        self.counts = {}  # empty container
        isFirst = 0
        pre = None
        combine = None
        with open(filePath, 'r') as infile:  # read file
            for line in infile:
                words = wordRE.findall(line)
                for word in words:
                    if isFirst == 0:
                        pre = word
                        isFirst += 1
                    else:
                        combine = pre + " " + word
                        pre = word
                        if combine not in self.counts:  # not in,set to 1
                            self.counts[combine] = 1
                        else:
                            self.counts[combine] += 1  # is in, accumulate
                        if combine not in self.weight:  # init weight to 0
                            self.weight[combine] = 0
                        combine = None
        self.container.append((self.counts, sentiment))

    def train_standard(self):  # train
        for dict, sen in self.train_data:
            self.score = 0
            for wd in dict:  # count the score
                self.score += self.weight[wd] * dict[wd]
            if self.score >= 0:  # predict as positive
                self.y_hat = 1
            else:  # negative
                self.y_hat = -1

            if self.y_hat != sen:  # update weight
                if sen == 1:  # if y == 1
                    self.weight[wd] += dict[wd]
                elif sen == -1:
                    self.weight[wd] -= dict[wd]

    def train_avg(self):  # train
        for dict, sen in self.train_data:
            self.score = 0
            for wd in dict:  # count score
                self.score += self.weight[wd] * dict[wd]
            if self.score >= 0:  # predict as positive
                self.y_hat = 1
            else:  # negative
                self.y_hat = -1
            if self.y_hat != sen:  # update weight
                for wd in dict:
                    self.weight[wd] += sen * dict[wd]  # sen = 1 or -1

    def count_avg_weight(self):  # average the weight
        for i in self.weight_c:
            for wd in self.weight_c[i]:
                self.weight[wd] += self.weight_c[i][wd]

    def shuffle(self, seed):  # shuffle the data
        random.seed(seed)  # to rebuild the result
        self.train_data = self.container[:1600]  # 1600 train
        self.valid_data = self.container[1600:]  # rest for validate
        random.shuffle(self.train_data)

    def fit(self, dict):  # make classification
        self.score = 0
        for wd in dict:
            self.score += self.weight[wd] * dict[wd]
        if self.score >= 0:
            return 1  # positive
        elif self.score < 0:
            return -1  # negative

    def eval(self,name):  # evaluate the result
        self.correct_validate = 0
        for dict, sen in self.valid_data:
            if self.fit(dict) == sen:  # correct classification
                self.correct_validate += 1
        self.result.append(self.correct_validate / len(self.valid_data))
        print("correctness_validate:-" + name, self.correct_validate / len(self.valid_data))

    def top10(self,name):#sort the weight and print top 10
        top = sorted(self.weight.items(), key=operator.itemgetter(1), reverse=True)
        print("top10 positive-" + name)
        print(top[:10])
        # print("top10 negative-" + name)
        # print(top[-10:])

    def progress(self, length,name):#plot the graph
        plt.plot(np.arange(length), self.result,label = name)
        plt.title('learning progress')
        plt.xlabel('iteration')
        plt.ylabel('precision')
        plt.legend(loc='best')
        # plt.title(name)
        
        
    def run(self):
        start1 = time.time()
        times = 25
        print("-------Reading file---Bag-of-words----------------")
        for i in cmd.train_n:  # init weight read the document
            self.bagOfWords(i, self.train_n[i])
        for i in cmd.train_p:
            self.bagOfWords(i, self.train_p[i])
        for i in cmd.fit_n:  # read data for fitting
            self.bagOfWords(i, self.fit_n[i])
        for i in cmd.fit_p:
            self.bagOfWords(i, self.fit_p[i])
        # self.weight_c[0] = self.weight.copy()  # set the init weight
        readTime1 = time.time()
        print("-------Training weight---Bag-of-words----------------")
        for iter in range(times):  # i to max iter
            self.shuffle(iter)
            self.train_avg()
            # cmd.count_avg_weight()
            print("iter: ", iter)
            self.eval("Bag of words")
        trainTime1 = time.time()
        self.top10("Bag of words")
        self.progress(times,"Bag of words")

        start2 = time.time()
        self.__init__()
        print("-------Reading file---Bigram----------------")
        for i in cmd.train_n:  # init weight read the document
            self.bigram(i, self.train_n[i])
        for i in cmd.train_p:
            self.bigram(i, self.train_p[i])
        for i in cmd.fit_n:  # read data for fitting
            self.bigram(i, self.fit_n[i])
        for i in cmd.fit_p:
            self.bigram(i, self.fit_p[i])
        # self.weight_c[0] = self.weight.copy()  # set the init weight
        readTime2 = time.time()

        
        print("-------Training weight---Bigram----------------")
        for iter in range(times):  # i to max iter
            self.shuffle(iter)
            self.train_avg()
            # cmd.count_avg_weight()
            print("iter: ", iter)
            self.eval("Bigram")
        trainTime2 = time.time()

        self.top10("Bigram")
        self.progress(times,"Bigram")

        print("Read Time-bag of words: ", readTime1 - start1)#print run time
        print("Train Time-bag of words: ", trainTime1 - readTime1)
        print("Read Time-Bigram: ", readTime2 - start2)
        print("Train Time-Bigram: ", trainTime2 - readTime2)
        plt.show()

if __name__ == '__main__':
    cmd = Perceptron()
    cmd.run()




