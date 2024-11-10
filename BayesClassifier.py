import numpy as np

class DataNode:
    def __init__(self, data, label):
        self.data = data
        self.label = label
    
    def __str__(self):
        return f'{self.data} {self.label}'
    
class BayesClassifier:
    def __init__(self):
        self.date = []
        
    def add_data(self, data_node):
        self.data.append(data_node)
        
