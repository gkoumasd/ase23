from torch.utils.data import Dataset

class Dataload(Dataset):
    
    def __init__(self, df):
        self.df = df
        self.file = df['File'].values
        self.category = df['Category'].values
        
    def __getitem__(self, item):
       
        file = self.file[item]
        category = self.category[item]
        
        return (file, category)
        
    def __len__ (self):
        return len(self.file)# -*- coding: utf-8 -*-