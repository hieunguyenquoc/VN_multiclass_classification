import pandas as pd 
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

class Preprocess:
    def __init__(self, args):
        self.data_path = "data/Data.csv"
        self.test_size = 0.2
        self.num_words = args.num_words
        self.max_len = args.max_len
    
    def remove_stopwords(strings):
        print("strings: " + strings)
        strings = strings.split()
        f = open("stopwords.txt", 'r', encoding="utf-8")  
        stopwords = f.readlines()
        stop_words = [s.replace("\n", '') for s in stopwords]
        doc_words = []
        #### YOUR CODE HERE ####
    
        for word in strings:
            if word not in stop_words:
                doc_words.append(word)

        #### END YOUR CODE #####
        doc_str = ' '.join(doc_words).strip()
        return doc_str
    
    def load_data(self):
        df = pd.read_csv(self.data_path, encoding="utf-8")
        
        text = df['texts']
        
        text_prepocess = []
        for i in text:
            i = self.remove_stopwords(i)
            i = i.lower()
            i = re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]',' ', i)
            i = re.sub(r'\s+', ' ', i).strip()
            text_prepocess.append(i)
        
        text_to_train = np.array(text_prepocess)

        label = df['labels'].lower().values
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(text_to_train, label, test_size=self.test_size)

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.Y_train)

        self.Y_train = self.label_encoder.transform(self.Y_train)
        self.Y_test = self.label_encoder.transform(self.Y_test)
    
    def Tokenization(self):
        self.tokenize = Tokenizer(num_words=self.num_words)
        self.tokenize.fit_on_texts(self.X_train)
    
    def sequence_to_text(self, input):
        sequence = self.tokenize.texts_to_sequences(input)
        return pad_sequences(sequence, maxlen=self.max_len)
        





