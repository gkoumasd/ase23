import os#
from tqdm import *
import random
import pandas as pd

def export_paths(path):
    
    seed_val = 42
    random.seed(seed_val)
    
    directory = path
    
    files_lst = []
    categories = []
    
    df = pd.DataFrame(columns = ['File', 'Category'])
    
    for dirpath , subdirs, files in os.walk(directory):
        for file in tqdm(files):
            if file.endswith(".rs"):
                files_lst.append(os.path.join(dirpath, file))
                categories.append(os.path.join(dirpath, file).split('/')[-2])
                
    df['File'] = files_lst
    df['Category'] = categories
    
    df.to_csv(os.path.join('data_train_test_val', path.split('/')[-1]+'.csv'),header=True)
            
    return files_lst, categories            
    
    
files, categories = export_paths('data_train_test_val/train')
files, categories = export_paths('data_train_test_val/test')
files, categories = export_paths('data_train_test_val/val')       