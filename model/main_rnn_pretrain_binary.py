import pandas as pd
import torch
import torch.nn as nn
import pickle
import sys
import cdsw

from torchtext import data
from sklearn.model_selection import train_test_split

data_dir = '/home/cdsw/data/'
model_dir = '/home/cdsw/model/'

sentiments = pd.read_csv(data_dir+'/Tweets.csv')
# use only not null text

clean_df = sentiments[sentiments['text'].notnull() &
                      sentiments['airline'].notnull() &
                      sentiments['airline_sentiment'].notnull() &
                      sentiments['tweet_id'].notnull()]
# use only tweet(text), airline, label (airline_sentiment) and tweet id
final_df = clean_df.filter(['tweet_id', 'text', 'airline',
                           'airline_sentiment'], axis=1)
# use only positive and negative sentiment
row_vals = ['positive', 'negative']
final_df = final_df.loc[final_df['airline_sentiment'].isin(row_vals)]
# use Delta only (this should be a toggle)
# final_df = final_df[final_df['airline'] == 'Delta']

# convert neutral, positive and negative to numeric
# sentiment_map = {'neutral': 0, 'positive': 1, 'negative': -1} 
# final_df['airline_sentiment'] = final_df['airline_sentiment'].map(sentiment_map)
# split into train, test, val (.7, .15, .15)
train_df, testval_df = train_test_split(final_df, test_size=0.3)
test_df, val_df = train_test_split(testval_df, test_size=0.5)

# convert df back to csv, with column names
train_df.to_csv(data_dir+'/train.csv', index=False)
test_df.to_csv(data_dir+'/test.csv', index=False)
val_df.to_csv(data_dir+'/val.csv', index=False)

# load into torchtext
ID = data.Field()
TEXT = data.Field(tokenize='spacy')
SENTIMENT = data.LabelField(dtype=torch.float)
AIRLINE = data.Field()

# access using batch.id, batch.text etc
fields = [('id', ID), ('text', TEXT), ('airline', AIRLINE), ('label', SENTIMENT)]
train_data, valid_data, test_data = data.TabularDataset.splits(path=data_dir,
                                                               train='train.csv',
                                                               validation='val.csv',
                                                               test='test.csv',
                                                               format='csv',
                                                               fields=fields,
                                                               skip_header=True)
# build iterators
MAX_VOCAB_SIZE = 10_000

ID.build_vocab(train_data)
# TEXT.build_vocab(train_data, max_size=MAX_VOCAB_SIZE)
TEXT.build_vocab(train_data,
                 max_size=MAX_VOCAB_SIZE,
                 vectors="glove.twitter.27B.25d",
                 unk_init=torch.Tensor.normal_)
SENTIMENT.build_vocab(train_data)
AIRLINE.build_vocab(train_data)

print(TEXT.vocab.freqs.most_common(20))
# save this - need for model prediction
outfile = open(model_dir+'vocab_index.pkl', 'wb')
pickle.dump(TEXT.vocab.stoi, outfile, -1)
outfile.close()
cdsw.track_file(model_dir+'vocab_index.pkl')

# check labels, 0 is negative, 1 is positive
print(SENTIMENT.vocab.stoi)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 32

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    sort_key=lambda x: x.text,  # sort by text
    batch_size=BATCH_SIZE,
    device=device)


# model
class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
       
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
       
    def forward(self, text):

        # text = [sent len, batch size]
        embedded = self.embedding(text)
        # embedded = [sent len, batch size, emb dim]
        output, hidden = self.rnn(embedded)
        # output = [sent len, batch size, hid dim]
        # hidden = [1, batch size, hid dim]
        assert torch.equal(output[-1, :, :], hidden.squeeze(0))

        return self.fc(hidden.squeeze(0)), hidden


INPUT_DIM = len(TEXT.vocab)
# EMBEDDING_DIM = 100
EMBEDDING_DIM = 25
HIDDEN_DIM = 256
OUTPUT_DIM = 1

model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

# copy pretrained into embedding layer

pretrained_embeddings = TEXT.vocab.vectors
print(pretrained_embeddings.shape)
model.embedding.weight.data.copy_(pretrained_embeddings)


import torch.optim as optim

#try learning rates 1e-2, 1e-3, 1e-4,1e-5 via Experiments
if len (sys.argv) == 2:
  if sys.argv[1].split(sep='=')[0]=='learning_rate' and isinstance(float(sys.argv[1].split(sep='=')[1]), float):
    learning_rate = float(sys.argv[1].split(sep='=')[1])
  else:
    sys.exit("Invalid Arguments passed to Experiment")
else:
    learning_rate = 1e-3

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()
model = model.to(device)
criterion = criterion.to(device)

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc

def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
        #predictions = model(batch.text).squeeze(1)
        predictions, _ = model(batch.text)
        predictions = predictions.squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):

    epoch_loss = 0
    epoch_acc = 0
   
    model.eval()
   
    with torch.no_grad():
   
        for batch in iterator:

            #predictions = model(batch.text).squeeze(1)
            predictions, _ = model(batch.text)
            predictions = predictions.squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 5

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()
    
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), model_dir+'/rnn_binary_pretrain_model.pt')
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

cdsw.track_file(model_dir+'/rnn_binary_pretrain_model.pt')

model.load_state_dict(torch.load(model_dir+'/rnn_binary_pretrain_model.pt'))

test_loss, test_acc = evaluate(model, test_iterator, criterion)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
cdsw.track_metric("Test Accuracy",round(train_acc, 2))


# expose this for api

import spacy
nlp = spacy.load('en')


def predict_sentiment(model, sentence):
    model.eval()
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    # print(tensor)
    sentiment, hidden = model(tensor)
    prediction = torch.sigmoid(sentiment)
    return prediction.item(), hidden


print(predict_sentiment(model, "you are horrible"))

# use with formatted train/val/test predict_sentiment_from_dataset(model, next(train_data.text))

def predict_sentiment_from_dataset(model, tokenized):
    model.eval()
    # tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    # print(tensor)
    sentiment, hidden = model(tensor)
    prediction = torch.sigmoid(sentiment)
    return prediction.item(), hidden


# save for front-end application
prediction_list = []
embedding_list = []
airline_list = []
tweet_list = []
for example in test_data:
    text = example.text  # this is tokenized
    airline = example.airline
    prediction, embedding = predict_sentiment_from_dataset(model, text)
    tweet_list.append(text)
    prediction_list.append(prediction)
    embedding_list.append(embedding.data.numpy().squeeze(1))
    airline_list.append(airline)

output_dict = {"prediction": prediction_list,
               "embedding": embedding_list,
               "tweet": tweet_list,
               "airline": airline_list}
outfile = open(data_dir+'frontend_data', 'wb')
pickle.dump(output_dict, outfile, -1)
outfile.close()
cdsw.track_file(data_dir+'frontend_data')

