# python3 lab5.py

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10

test_sentence = ["The mathematician ran","The mathematician ran to the store","The physicist ran to the store","The philosopher thought about it","The mathematician solved the open problem"]
test_sentences = [i.split() for i in test_sentence]
for i in range(len(test_sentences)):
    test_sentences[i] = ["START"] + test_sentences[i] + ["END"]

trigrams = []
for i in range(len(test_sentences)):
    for j in range(len(test_sentences[i])-2):
        trigrams += [([test_sentences[i][j], test_sentences[i][j + 1]], test_sentences[i][j + 2])]


vocab = {}# wd: tf
for i in range(len(test_sentences)):
    for j in range(len(test_sentences[i])):
        if test_sentences[i][j] not in vocab:
            vocab[test_sentences[i][j]] = 1
        else:
            vocab[test_sentences[i][j]] += 1

word_to_ix = {word: i for i, word in enumerate(vocab)} #mapping index : word

class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

#Sanity check
print("------Running Sanity check------")
check = True
for i in range(5):# 5 consecutive runs
    losses = []
    loss_function = nn.NLLLoss()
    model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(500):
        total_loss = torch.Tensor([0])
        for context, target in trigrams:
            # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
            # into integer indices and wrap them in variables)
            context_idxs = [word_to_ix[w] for w in context]
            context_var = autograd.Variable(torch.LongTensor(context_idxs))
            # print(context_idxs,context_var) #id number of first two words

            # Step 2. Recall that torch *accumulates* gradients. Before passing in a
            # new instance, you need to zero out the gradients from the old
            # instance
            model.zero_grad()

            # Step 3. Run the forward pass, getting log probabilities over next
            # words
            log_probs = model(context_var)
            # Step 4. Compute your loss function. (Again, Torch wants the target
            # word wrapped in a variable)
            loss = loss_function(log_probs, autograd.Variable(
                torch.LongTensor([word_to_ix[target]])))
            # Step 5. Do the backward pass and update the gradient
            loss.backward()
            optimizer.step()

            total_loss += loss.data
        losses.append(total_loss)
        # # print(losses)  # The loss decreased every iteration over the training data!

    test = "START The mathematician ran to the store END".split()
    trigram_test = [([test[i], test[i + 1]], test[i + 2])
                    for i in range(len(test) - 2)]

    for context, target in trigram_test:
        context_idxs = [word_to_ix[w] for w in context]
        context_var = autograd.Variable(torch.LongTensor(context_idxs))
        log_probs = model(context_var)
        y_pre = log_probs.argmax(1)
        y = word_to_ix[target]
        if y != y_pre:
            check = False
            print("Sanity check Failed! ")

if check:
    print("Sanity check Passed!")
else:
    print("Sanity check Failed!")


#predict for the context “START The” the word “mathematician”
context_idxs = [word_to_ix["START"],word_to_ix["The"]]
context_var = autograd.Variable(torch.LongTensor(context_idxs))
log_probs = model(context_var)
print("The probability of \"mathematician\": ", log_probs.data[0][word_to_ix["mathematician"]])
print("The probability of \"physicist\": ", log_probs.data[0][word_to_ix["physicist"]])

#Test
print("-----Running Test : The ______ solved the open problem.-------")
probability = []#[phy,phi]

test_sent = [["START", "The", "physicist", "solved", "the", "open", "problem"], ["START", "The", "philosopher", "solved", "the", "open", "problem"]]
for sent in test_sent:
    for i in range(len(test_sentences)):
        for j in range(len(test_sentences[i]) - 2):
            trigram_test += [([test_sentences[i][j], test_sentences[i][j + 1]], test_sentences[i][j + 2])]
    prob = 1
    for context, target in trigram_test:
        context_idxs = [word_to_ix[w] for w in context]
        context_var = autograd.Variable(torch.LongTensor(context_idxs))
        log_probs = model(context_var)
        target_idx = word_to_ix[target]
        prob *= log_probs.data[0][target_idx]
    probability.append(prob)

if probability[0] > probability[1]:
    print("Predict: The physicist solved the open problem.")
else:
    print("Predict: The philosopher solved the open problem.")

#embeddings similarity of "physicist" and "mathematician" / "philosopher" and "mathematician"
print("-----Running Embedding similarity-----")

embeds = model.embeddings
lookup_tensor = torch.tensor([word_to_ix["philosopher"]], dtype=torch.long)
phi_embed = embeds(lookup_tensor)
lookup_tensor = torch.tensor([word_to_ix["mathematician"]], dtype=torch.long)
mat_embed = embeds(lookup_tensor)
lookup_tensor = torch.tensor([word_to_ix["physicist"]], dtype=torch.long)
phy_embed = embeds(lookup_tensor)

cos = nn.CosineSimilarity()
simPhyMath = cos(phy_embed, mat_embed)
simPhiMath = cos(phi_embed, mat_embed)
print("The embeddings for “physicist” and “mathematician” similarity : ",simPhyMath )
print("The embeddings for “philosopher” and “mathematician” similarity: ",simPhiMath)






