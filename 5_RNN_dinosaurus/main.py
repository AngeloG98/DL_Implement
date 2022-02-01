from dinos import read_dinos
from rnn import RNN

if __name__ == "__main__":
    char_to_ix, ix_to_char, train_words = read_dinos('0_dataset/Dinosaurs/dinos.txt')
    hidden_size = 100
    in_out_size = len(char_to_ix)
    rnn = RNN(100, in_out_size, in_out_size)
    rnn.train(train_words, ix_to_char, char_to_ix, batchsize=32, lr=0.1, epochs=30)