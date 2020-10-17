import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

class EncoderRNN(nn.Module):

    def __init__(self, embedding, emb_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size // 2 # use bidirectional RNN
        self.embedding = embedding
        self.LSTM = nn.LSTM(input_size=emb_size,
                            hidden_size=self.hidden_size,
                            num_layers=2,
                            batch_first=True,
                            dropout=0.2,
                            bidirectional=True)

        return

    def forward(self, inputs, input_lengths):
        """forward path, note that inputs are batch first"""

        embedded = self.embedding(inputs)
        lengths_list = input_lengths.view(-1).tolist()
        packed_emb = pack(embedded, batch_first=True, lengths=lengths_list, enforce_sorted=False)

        memory_bank, encoder_final = self.LSTM(packed_emb)
        memory_bank = unpack(memory_bank)[0].view(inputs.size(0), inputs.size(1), -1)

        return memory_bank, encoder_final