import os#
from tqdm import *
import random
from sklearn.model_selection import train_test_split
import shutil


def split_data(category):
    
    seed_val = 42
    random.seed(seed_val)

    directory = 'data/' + category
    files_lst = []
    
    for dirpath , subdirs, files in os.walk(directory):
        for file in tqdm(files):
            if file.endswith(".rs"):
                files_lst.append(os.path.join(dirpath, file))
    
        
    random.shuffle(files_lst)
    
    #Sample the safe function to tackle the high imbalance
    if category == 'safe':
        files_lst = files_lst[:int(len(files_lst)*0.25)]
    elif category == 'unsafe':
        files_lst = files_lst[:int(len(files_lst)*0.25)]    
        
    
    print('Total files for the %s category:%d'%(category,len(files_lst)))   
 
    
    
    # In the first step we will split the data in training and remaining dataset
    
    X_train, X_rem = train_test_split(files_lst, train_size=0.8, random_state=42)
   
    # Now since we want the valid and test size to be equal (10% each of overall data). 
    # we have to define valid_size=0.5 (that is 50% of remaining data)
    test_size = 0.5
    X_valid, X_test = train_test_split(X_rem, test_size=0.5, random_state=42)
    
    if category == 'safe':
        shutil.rmtree('data_train_test_val', ignore_errors=True)
        os.mkdir('data_train_test_val')
        os.mkdir('data_train_test_val/train')
        os.mkdir('data_train_test_val/train/safe')
        os.mkdir('data_train_test_val/val')
        os.mkdir('data_train_test_val/val/safe')
        os.mkdir('data_train_test_val/test')
        os.mkdir('data_train_test_val/test/safe')
    elif category == 'unsafe':       
        os.mkdir('data_train_test_val/train/unsafe')
        os.mkdir('data_train_test_val/val/unsafe')
        os.mkdir('data_train_test_val/test/unsafe')
    
    for file in X_train:
        shutil.copy2(file, 'data_train_test_val/train/' + category)
    
    for file in X_test:
        shutil.copy2(file, 'data_train_test_val/test/' + category)  
        
    for file in X_valid:
        shutil.copy2(file, 'data_train_test_val/val/' + category)  
        
        
#split_data('safe')
split_data('unsafe')        