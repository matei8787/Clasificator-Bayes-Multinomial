import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
import csv

matplotlib.use('Qt5Agg') 
CARACTERE_SPECIALE = ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '+', '-', '=', '{', '}', '[', ']', '|', '\\', ':', ';', '"', '<', '>', ',', '.', '?', '/', '~', '`']


class DataNode:
    def __init__(self, data, label):
        self.data = data
        self.label = label
    
    def __str__(self):
        return '{} : {}'.format(self.label, self.data)

class BayesClassifier:
    #datele sunt tinute in date[label] = {data1: count1, data2: count2} ...
    #probabilitatile sunt tinute in probabilitati[label] = {cuvant1: probabilitate1, cuvant2: probabilitate2} ...
    def __init__(self):
        self.date = []
        self.X_train = []
        self.y_train = []
        self.probabilitati = {}
        
    def add_data(self, data_node):
        self.date.append(data_node)
        
    def print_date(self, file=None):
        if file:
            with open(file, 'w', encoding='latin-1') as f:
                f.write('Train data:\n')
                for label in self.date:
                    for key, val in self.date[label].items():
                        f.write(str(key))
                        f.write(' ')
                        f.write(str(val))
                        f.write('\n')
        else:
            for data_node in self.date:
                print(data_node)
                print('\n')
                
    #impartim datele in date de antrenament si date de test
    def split_date(self):
        x = [i.data for i in self.date]
        y = [i.label for i in self.date]
        self.X_train, X_test, self.y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=69)
        return X_test, y_test
        
        
    TOTAL_LABEL = '$total#label$'
    #Preparam frumos datele
    def init_date(self):
        for i in range(len(self.X_train)):
            for char in CARACTERE_SPECIALE:
                self.X_train[i] = str(self.X_train[i]).replace(char, ' ')
        self.X_train = [data.split() for data in self.X_train]
        self.X_train = [DataNode(data, label) for data, label in zip(self.X_train, self.y_train)]
        self.date = {}
        for data_node in self.X_train:
            if data_node.label not in self.date:
                self.date[data_node.label] = {}
                self.date[data_node.label][self.TOTAL_LABEL] = 0
            for word in data_node.data:
                if word not in self.date[data_node.label]:
                    self.date[data_node.label][word] = 1
                else:
                    self.date[data_node.label][word] += 1
                self.date[data_node.label][self.TOTAL_LABEL] += 1
                
    def print_date_impartite(self, file=None):
        if file:
            with open(file, 'w') as f:
                f.write('Train data:\n')
                f.write(str(self.date))
        else:
            print('Train data:')
            print(self.date)
            
    
    def vizualizare_date_label(self):
        plt.hist(self.y_train, bins=2)
        plt.show()
    
    def update_cuvant(self, cuv, label):
        if label not in self.date:
            self.date[label] = {}
            self.date[label][self.TOTAL_LABEL] = 0
        if cuv in self.date[label]:
            self.date[label][cuv] += 1
            self.date[label][self.TOTAL_LABEL] += 1
            self.probabilitati[label][cuv] = (self.date[label][cuv] + 1) / (self.date[label][self.TOTAL_LABEL] + len(self.date[label]) - 1)
        else:
            self.date[label][cuv] = 1
            self.date[label][self.TOTAL_LABEL] += 1
            self.probabilitati[label][cuv] = (self.date[label][cuv] + 1) / (self.date[label][self.TOTAL_LABEL] + len(self.date[label]) - 1)
    
    def train(self):
        for label in self.date:
            self.probabilitati[label] = {}
            for cuv, nrap in self.date[label].items():
                if cuv == self.TOTAL_LABEL:
                    continue
                self.probabilitati[label][cuv] = (nrap + 1) / (self.date[label][self.TOTAL_LABEL] + len(self.date[label]) - 1)

    def predict(self, data):
        #data este practic un sir de caractere cu multe cuvinte
        #vrem sa impartim data in cuvinte dupa CARACTERE_SPECIALE
        for char in CARACTERE_SPECIALE:
            data = data.replace(char, ' ')
        data = data.split()
        
        probabilitate_maxima = -1
        label_maxima = None
        for label in self.date:
            probabilitate = 1
            for cuv in data:
                if cuv in self.probabilitati[label]:
                    #self.update_cuvant(cuv, label)
                    probabilitate *= self.probabilitati[label][cuv]
                    #self.update_cuvant(cuv, label)
                else:
                    #self.update_cuvant(cuv, label)
                    probabilitate *= 1 / (self.date[label][self.TOTAL_LABEL] + len(self.date[label]) - 1)
                    #self.update_cuvant(cuv, label)
            if probabilitate > probabilitate_maxima:
                probabilitate_maxima = probabilitate
                label_maxima = label
            
        return label_maxima
    



bc = BayesClassifier()

def make_date_mici():
    with open('./date/spam_mic.csv', 'w', newline='', encoding='latin-1') as f:
        writer = csv.writer(f)
        writer.writerow(['v1', 'v2'])
        writer.writerow(['spam', 'buy buy buy buy buy buy buy buy buy buy buy buy buy buy buy buy buy buy buy buy'])
        writer.writerow(['spam', 'now now now now now'])
        writer.writerow(['spam', 'free free free free free free free free free free'])
        writer.writerow(['ham', 'buy buy buy buy buy'])
        writer.writerow(['ham', 'now now now now now now now now now now now now now now now'])
        writer.writerow(['ham','free free free free free'])
        

def load_data(file, encoding='latin-1', label_row='v1', data_row='v2', delimiter=','):
    data = pd.read_csv(file, encoding=encoding, delimiter=delimiter)
    for _, row in data.iterrows():
        aux = DataNode(row[data_row], row[label_row])
        bc.add_data(aux)    

def main():
    #make_date_mici()
    file = input("Introduceti numele fisierului: ")
    if file.lower() == 'spam_mic':
        file='./date/spam_mic.csv'
        encoding = 'latin-1'
        label_row = 'v1'
        data_row = 'v2'
    elif file.lower() == 'spam':
        file='./date/spam.csv'
        encoding = 'latin-1'
        label_row = 'v1'
        data_row = 'v2'
    elif file.lower() == 'politic':
        file='./date/dataset_politic.csv'
        encoding = 'latin-1'
        label_row = 'Emotion'
        data_row = 'Text_of_Speech'
    elif file.lower() == 'emotie':
        file='./date/emotion_dataset_raw.csv'
        encoding = 'latin-1'
        label_row = 'Emotion'
        data_row = 'Text'
    else:
        print('Usage: spam_mic, spam, politic, emotie')
    load_data(file, encoding=encoding, label_row=label_row, data_row=data_row)
    X_test, y_test = bc.split_date()
    bc.init_date()
    bc.train()
    bune = 0
    total = 0
    for data, label in zip(X_test, y_test):
        predictie = bc.predict(data)
        if predictie == label:
            bune += 1
        total += 1
        print(predictie, ' ', label)
    print('bune: ', bune)
    print('total: ', total)
    print('Probabilitate sa fie bun (acuratete): ', bune / total)

if __name__ == '__main__':
    main()
    
