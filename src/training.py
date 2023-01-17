import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from model import VN_Text_Classification
from preprocess import Preprocess
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from parser_param import parameter_parser

class DataMapper(Dataset):
    def __init__(self, x, y):
        
        self.x = x
        self.y = y 
    
    def __len__(self,x):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class Execute:
    def __init__(self,args):
        self.__init_data__(args)

        self.batch_size = args.batch_size
        self.model = VN_Text_Classification(args)

    def __init_data__(self,args):
        self.preprocess = Preprocess(args)
        self.preprocess.load_data()
        self.preprocess.Tokenization()

        raw_data_train = self.preprocess.X_train
        raw_data_test = self.preprocess.X_test

        self.x_train = self.preprocess.sequence_to_text(raw_data_train)
        self.x_test = self.preprocess.sequence_to_text(raw_data_test)

        self.y_train = self.preprocess.Y_train
        self.y_test = self.preprocess.Y_test
    
    def train(self):
        train = DataMapper(self.x_train, self.y_train)
        test = DataMapper(self.x_test, self.y_test)

        self.load_train = DataLoader(train, batch_size = self.batch_size)
        self.load_test = DataLoader(test, batch_size = self.batch_size)

        optimizer = optim.Adam(self.model.parameters(), lr = args.learning_rate)

        print(self.model)

        train_loss = []
        val_loss = []    

        for epoch in range(args.epochs):
            self.model.train()
            avg_loss = 0

            for x_batch, y_batch in self.load_train:
                x = x_batch.type(torch.LongTensor)
                y = y_batch.type(torch.FloatTensor).unsqueeze()

                y_pred = self.model(x)

                loss = F.cross_entropy(y_pred, y)

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                avg_loss += loss.item() / len(self.load_train)

            self.model.eval()   
            avg_val_loss = 0
            test_pred = np.zeros((len(self.x_test), len(self.preprocess.label_encoder)))

            with torch.no_grad():
                for i, (x_batch, y_batch) in enumerate(self.load_test()):
                    x = x_batch.type(torch.LongTensor)
                    y = y_batch.type(torch.FloatTensor)

                    y_pred = self.model(x)

                    avg_val_loss += F.cross_entropy(y_pred, y).item() / len(self.load_test) 
                    test_pred[i * args.batch_size:(i+1) * args.batch_size] = F.softmax(y_pred).numpy()

                test_accuracy = sum(test_pred.argmax(axis=1)==self.preprocess.Y_test)/len(self.preprocess.Y_test)
                train_loss.append(avg_loss)
                val_loss.append(avg_val_loss)
                print('Epoch {}/{} \t loss={:.4f} \t test_loss={:.4f}  \t test_acc={:.4f}'.format(
                epoch + 1, args.epochs, avg_loss, avg_val_loss, test_accuracy))

        torch.save(self.model, "model/viet_nam_classification.ckpt")

if __name__ == "__main__":
    args = parameter_parser()
    excute = Execute(args)
    excute.train


