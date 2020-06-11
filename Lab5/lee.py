import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
# Training data set from Lab 5
data = ["START The mathematician ran END".split(),
        "START The mathematician ran to the store END".split(),
        "START The physicist ran to the store END".split(),
        "START The philosopher thought about it END".split(),
        "START The mathematician solved the open problem END".split()]

# we should tokenize the input, but we will ignore that for now
# build a list of tuples.  Each tuple is ([ word_i-2, word_i-1 ], target word)
trigrams=[]
for sentence in data:
    trigram = [([sentence[i], sentence[i + 1]], sentence[i + 2])
                for i in range(len(sentence) - 2)]
    trigrams.append(trigram)


# define word_to_ix dictionary to store the index of each word
word_to_ix = {}
for sent in data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

print(word_to_ix)

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

#-------------------------------------------
#         Run a Sanity check
#-------------------------------------------
# 5 consecutive runs
check = True
for i in range(5):
    #--------------------------train-------------------------
    losses = []
    loss_function = nn.NLLLoss()
    model = NGramLanguageModeler(len(word_to_ix), EMBEDDING_DIM, CONTEXT_SIZE)
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(800):
        total_loss = torch.Tensor([0])
        for sent_trigram in trigrams:
            for context, target in sent_trigram:

                # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
                # into integer indices and wrap them in variables)
                context_idxs = [word_to_ix[w] for w in context]
                context_var = autograd.Variable(torch.LongTensor(context_idxs))

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
    #print(losses)  # The loss decreased every iteration over the training data!

    #----------------------test---------------------------
    # test sentence: "The mathematician ran to the store"
    # lr=0.001 epoch=800
    test = data[1]
    trigram_test = trigrams[1]

    for context,target in trigram_test:
        context_idxs = [word_to_ix[w] for w in context]
        context_var = autograd.Variable(torch.LongTensor(context_idxs))
        log_probs = model(context_var)
        y_pre = log_probs.argmax(1)
        y = word_to_ix[target]
        if y!=y_pre:
            check = False
            print("Sanity check is Wrong ")
# if check:
#     print("Pass sanity check!!")
#
# #predict for the context “START The” the word “mathematician”
# context_idxs = [word_to_ix["START"],word_to_ix["The"]]
# context_var = autograd.Variable(torch.LongTensor(context_idxs))
# log_probs = model(context_var)
# target_idx = word_to_ix["mathematician"]
# physicist_idx = word_to_ix["physicist"]
# y_pre = log_probs.argmax(1)
# y = target_idx
# if y == y_pre:
#     print(" Pass context \"START The\" test. The predict word is \"mathematician\"")
# print("The probability of \"mathematician\": ", log_probs.data[0][target_idx])
# print("The probability of \"physicist\": ", log_probs.data[0][physicist_idx])
#
# #-------------------------------------------
# #         Test
# #-------------------------------------------
# #train
# torch.manual_seed(1)
# losses = []
# loss_function = nn.NLLLoss()
# model = NGramLanguageModeler(len(word_to_ix), EMBEDDING_DIM, CONTEXT_SIZE)
# optimizer = optim.SGD(model.parameters(), lr=0.001)
# for epoch in range(800):
#     total_loss = torch.Tensor([0])
#     for sent_trigram in trigrams:
#         for context, target in sent_trigram:
#
#             # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
#             # into integer indices and wrap them in variables)
#             context_idxs = [word_to_ix[w] for w in context]
#             context_var = autograd.Variable(torch.LongTensor(context_idxs))
#
#             # Step 2. Recall that torch *accumulates* gradients. Before passing in a
#             # new instance, you need to zero out the gradients from the old
#             # instance
#             model.zero_grad()
#
#             # Step 3. Run the forward pass, getting log probabilities over next
#             # words
#             log_probs = model(context_var)
#
#             # Step 4. Compute your loss function. (Again, Torch wants the target
#             # word wrapped in a variable)
#             loss = loss_function(log_probs, autograd.Variable(
#                 torch.LongTensor([word_to_ix[target]])))
#
#             # Step 5. Do the backward pass and update the gradient
#             loss.backward()
#             optimizer.step()
#
#             total_loss += loss.data
#     losses.append(total_loss)
# # print(losses)  # The loss decreased every iteration over the training data!
#
# # Q: The ______ solved the open problem ： “physicist” or “philosopher”?
# # calculate the probabilities of all the words in the sentence
# # P(x) =∏P(x_n|x_{n-2} x_{n-1})
# test_sent_phy = ["START", "The", "physicist", "solved", "the", "open", "problem"]
# test_sent_phi = ["START", "The", "philosopher", "solved", "the", "open", "problem"]
# test_sent = [test_sent_phy, test_sent_phi]
# pro_phy_phi = []
# for sent in test_sent:
#     trigram_test = [([sent[i], sent[i + 1]], sent[i + 2])
#                     for i in range(len(sent) - 2)]
#     pro = 1
#     for context, target in trigram_test:
#         context_idxs = [word_to_ix[w] for w in context]
#         context_var = autograd.Variable(torch.LongTensor(context_idxs))
#         log_probs = model(context_var)
#         target_idx = word_to_ix[target]
#         pro *= log_probs.data[0][target_idx]
#     pro_phy_phi.append(pro)
# pro_ans_phy = pro_phy_phi[0]
# pro_ans_phi = pro_phy_phi[1]
# print("Probabily: The ___physicist___ solved the open problem. ", pro_ans_phy)
# print("Probabily: The ___philosopher___ solved the open problem. ", pro_ans_phi)
#
# # get the vector of word "physicist" "mathematician" "philosopher"
# embeds = model.embeddings
# lookup_tensor = torch.tensor([word_to_ix["physicist"]], dtype=torch.long)
# physicist_embed = embeds(lookup_tensor)
# lookup_tensor = torch.tensor([word_to_ix["mathematician"]], dtype=torch.long)
# mathematician_embed = embeds(lookup_tensor)
# lookup_tensor = torch.tensor([word_to_ix["philosopher"]], dtype=torch.long)
# philosopher_embed = embeds(lookup_tensor)
#
# # calculate the similarity between "physicist" and "mathematician"
# cos = nn.CosineSimilarity()
# sim_phy_math = cos(physicist_embed, mathematician_embed)
# # calculate the similarity between "philosopher" and "mathematician"
# sim_phi_math = cos(philosopher_embed, mathematician_embed)
#
# print("similarity of \"physicist\" and \"mathematician\": ", sim_phy_math)
# print("similarity of \"philosopher\" and \"mathematician\": ", sim_phi_math)