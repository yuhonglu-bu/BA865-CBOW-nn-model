# Data Cleaning
import pandas as pd
import string

def clean_text(text):    
    x = text.translate(str.maketrans('', '', string.punctuation)) # remove punctuation
    x = x.lower().split() # lower case and split by whitespace to differentiate words
    return x

example_text = pd.read_csv('https://raw.githubusercontent.com/dylanwalker/BA865/master/datasets/hw3.csv')
cleaned_text = example_text.Review[:100].apply(clean_text)

#Create vocab & word_to_index
vocab = set()
for review in cleaned_text:
  vocab.update(set(review))
word_to_index = {word: i for i, word in enumerate(vocab)}

# Define make_cbow_data function 
cbow_data = []
def make_cbow_data(text, window_size):
  for text in cleaned_text:
    for i in range(window_size, len(text) - window_size):
      target = text[i]
      context_index = list(range(i - window_size, i + window_size + 1))
      context_index.remove(i)
      context = []
      for index in context_index:
        context.append(text[index])
      cbow_data.append((context, target))

# Define your CBOW model here
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional

class CBOW(nn.Module):
  def __init__(self, vocab_size, embed_dim, window_size, hidden_dim):
    super(CBOW, self).__init__()
    self.embedding = nn.Embedding(vocab_size, embed_dim)
    self.fc1 = nn.Linear(2 * window_size * embed_dim, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, vocab_size)

  def forward(self, x):
    x = self.embedding(x)
    x = x.view(1, -1)
    x = self.fc1(x)
    x = nn.functional.relu(x, inplace = True)
    x = self.fc2(x)
    x = nn.functional.log_softmax(x)
    return x

# Train model
# Parameters
VOCAB_SIZE = len(vocab)
EMBED_DIM = 100
WINDOW_SIZE = 2
HIDDEN_DIM = 30
N_EPOCHS = 300

# Trainning
make_cbow_data(cleaned_text, WINDOW_SIZE)
cbow_model = CBOW(VOCAB_SIZE, EMBED_DIM, WINDOW_SIZE, HIDDEN_DIM)
if torch.cuda.is_available():
    cbow_model = cbow_model.cuda()
    
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cbow_model.parameters(), lr = 0.001)
loss_data = []
for epoch in range(300):
    running_loss = 0
    for word in cbow_data:
        context, target = word
        context = Variable(torch.LongTensor([word_to_index[i] for i in context]))
        target = Variable(torch.LongTensor([word_to_index[target]]))
        if torch.cuda.is_available():
            context = context.cuda()
            target = target.cuda()
        output = cbow_model(context)
        loss = criterion(output, target)
        running_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss_epoch = running_loss / len(cbow_data)
    loss_data.append(loss_epoch)

## Plot losses by epochs
import matplotlib.pyplot as plt 
epoch = list(range(1, 301))
plt.plot(epoch, loss_data)
plt.xlabel('Number of epochs')
plt.ylabel('Loss')
plt.show()

## Print five most similar words with the word "delicious"

from math import sqrt as sqrt
from pandas.core.frame import DataFrame
embed = cbow_model.embedding.weight.data.cpu().numpy()

def CosDistance(a, b):
  mul=0
  ma=0
  mb=0
  for i in range(len(a)):
    mul += a[i]*b[i]
    ma += a[i]*a[i]
    mb += b[i]*b[i]
  cos = mul/sqrt(ma*mb)
  return cos

cos_dis_embed = []
for word in embed:
  cos_dis = CosDistance(embed[word_to_index['delicious']], word)
  cos_dis_embed.append(cos_dis)
cos_dis_index = list(range(0, len(cos_dis_embed)))
cosine_distance = dict(zip(cos_dis_index, cos_dis_embed))
cosine_distance_sort = DataFrame(sorted(cosine_distance.items(), key = lambda item:item[1], reverse = True))
top5_index = cosine_distance_sort[0][1:6]
for index in range(5):
  print('The top', index + 1, 'synonyms is', list(word_to_index.keys())[list(word_to_index.values()).index(top5_index[index + 1])])