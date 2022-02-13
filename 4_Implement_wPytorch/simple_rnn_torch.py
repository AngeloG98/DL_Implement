from turtle import forward
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

def read_dinos(filename):
    #load vocab
    data = open(filename, 'r').read()
    data = data.lower()
    chars = list(set(data))
    char_to_ix = { ch:i for i,ch in enumerate(sorted(chars)) }
    ix_to_char = { i:ch for i,ch in enumerate(sorted(chars)) }
    #load words
    with open(filename) as f:
        words = f.readlines()
    words = [x.lower().strip() for x in words]

    return char_to_ix, ix_to_char, words

class RNN(nn.Module):
    def __init__(self, i_size, h_size, o_size) -> None:
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size = i_size, hidden_size = h_size, num_layers = 1)
        self.fc = nn.Linear(h_size, o_size)
    
    def forward(self, x, h):
        out, h = self.rnn(x, h)
        k = self.fc(out)
        y_pred = F.softmax(self.fc(out), dim=2)
        return y_pred, h

def train(model, train_data, char_to_ix, optimizer, epoch, device, logging):
    idx = 0
    for data in train_data:
        # init
        optimizer.zero_grad()
        h_state = None # Defaults to zeros if not provided.
        # load data
        X = [None] + [char_to_ix[ch] for ch in data]
        Y = X[1:] + [char_to_ix["\n"]]
        seq_len = len(X)
        in_out_size = len(char_to_ix)
        input = np.zeros((seq_len, 1, in_out_size)) # (time_step, batch, input_size)
        target = np.zeros((seq_len, 1))
        for i in range(seq_len):
            input[i,0,:] = np.zeros(in_out_size)
            input[i,0, X[i]] = 1
            target[i,0] = Y[i]
        input = input.astype(np.float32)
        target = target.astype(np.float32)
        input = torch.from_numpy(input).to(device)
        target = torch.from_numpy(target).to(device)
        # train
        y_pred, h_state = model(input, h_state)
        h_state = h_state.detach()
        y_pred = torch.log(y_pred)
        loss = F.nll_loss(y_pred.transpose(1,0).transpose(2,1), target.transpose(1,0).type(torch.LongTensor).to(device))
        loss.backward()
        optimizer.step()
        # print
        if idx%logging == 0:
            print("epoch:{}, idx:{}, loss:{}".format(epoch, idx, loss.item()))
        idx += 1

def predict(model, char_to_ix, ix_to_char, seed):
    indices = []
    idx = -1 
    counter = 0
    newline_character = char_to_ix['\n']
    in_size = len(char_to_ix)
    h_state = None
    x = np.zeros((1, 1, in_size)).astype(np.float32)
    x = torch.from_numpy(x).to(device)
    while (idx != newline_character and counter != 50):
        y_pred, h_state = model(x, h_state)
        np.random.seed(counter + seed)
        y_pred = y_pred.detach().cpu().numpy() # :( not a good way
        idx = np.random.choice(range(y_pred.shape[2]),p=y_pred.ravel())
        indices.append(idx)

        x = np.zeros((1, 1, in_size)).astype(np.float32)
        x[0, 0, idx] = 1
        x = torch.from_numpy(x).to(device)
        # update seed
        seed += 1
        counter +=1
    if (counter == 50):
        indices.append(char_to_ix['\n'])

    indices_char = ''
    for i in indices:
        indices_char += ix_to_char[i]

    return indices, indices_char[:-1]


if __name__ == '__main__': 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    char_to_ix, ix_to_char, train_words = read_dinos('0_dataset/Dinosaurs/dinos.txt')
    hidden_size = 100
    in_out_size = len(char_to_ix)
    model = RNN(in_out_size, hidden_size, in_out_size).to(device)

    lr = 0.1
    optimizer = optim.Adagrad(model.parameters(), lr=lr)

    epochs = 50
    logging = 100
    for epoch in range(epochs):
        train(model, train_words, char_to_ix, optimizer, epoch, device, logging)
        seed = 10 + epoch
        num = 5
        for i in range(num):
            indices, indices_char = predict(model, char_to_ix, ix_to_char, seed)
            print("predict: ", indices_char)
            seed += 1
        
        
        