import torch
import torch.nn as nn

class VN_Text_Classification(nn.Module):
    def __init__(self,args):
        super(VN_Text_Classification, self).__init__()
        
        self.input_size = args.num_words
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.num_class = 10
        self.dropout= nn.Dropout(0.5)

        self.embedd = nn.Embedding(self.input_size, self.hidden_dim, padding_idx=0)
        self.RNN = nn.RNN(input_size = self.hidden_dim, hidden_size = self.hidden_dim, num_layers = self.num_layers, batch_first = True)
        self.fc1 = nn.Linear(in_features=self.hidden_dim, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=self.num_class)
    
    def forward(self, x):
        h = torch.zeros((self.num_layers, x.size(0), self.hidden_dim))

        out = self.embedd(x)

        out, hidden = self.RNN(out, h)

        out = self.dropout(out)

        out = torch.relu_(self.fc1[out[:,-1,:]])

        out = self.fc2(out)

        out = self.fc3(out)

        return out