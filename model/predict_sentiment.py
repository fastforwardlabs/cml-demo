import torch
import torch.nn as nn
import spacy
import json
import pickle
import os
nlp = spacy.load('en')

model_dir = '/home/cdsw/model/'

#first look for vocab that came from experiments
if os.path.exists('/home/cdsw/vocab_index.pkl'):
  vocab_file_path = '/home/cdsw/vocab_index.pkl' 
else:
  vocab_file_path = model_dir+'/vocab_index.pkl'

#first look for model that came from experiments
if os.path.exists('/home/cdsw/rnn_binary_pretrain_model.pt'):
  model_file_path = '/home/cdsw/rnn_binary_pretrain_model.pt'
else:
  model_file_path = model_dir+'/rnn_binary_pretrain_model.pt'


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


INPUT_DIM = 10002
# EMBEDDING_DIM = 100
EMBEDDING_DIM = 25
HIDDEN_DIM = 256
OUTPUT_DIM = 1
vocab_index = pickle.load(open(vocab_file_path, 'rb'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
model.load_state_dict(torch.load(model_file_path))


def predict_sentiment_get_embedding(args):
    sentence = args.get('sentence')
    model.eval()
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [vocab_index[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    # print(tensor)
    sentiment, hidden = model(tensor)
    prediction = torch.sigmoid(sentiment)
    return json.dumps({'sentiment': prediction.item(),
                       'embedding': hidden.data.tolist()})

def predict_sentiment(args):
    sentence = args.get('sentence')
    model.eval()
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [vocab_index[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    # print(tensor)
    sentiment, hidden = model(tensor)
    prediction = torch.sigmoid(sentiment)
    return round(prediction.item(), 5)
  
  
#test
#x={"sentence" : "you are horrible"}
#predict_sentiment(x)

