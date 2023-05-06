import os
import torch
import torch.nn as nn
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score
from sklearn.metrics import classification_report
from argument_parser import parse_arguments
from transformers import get_linear_schedule_with_warmup
import pandas as pd
from torch.utils.data import DataLoader
import time
from random import shuffle
from models.T5_model import T5_model
from data_loader_T5 import Dataload
from transformers import RobertaTokenizer
# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


def train(opt):
    label_names = ["safe","unsafe"]
    
    df = pd.DataFrame(columns = ['Epoch','Tr Loss', 'Val Loss', 'Tr Acc', 'Val Acc',  'Time','lr'])
    model_bin = [opt.task, opt.model_name, opt.batch_size, opt.learning_rate]
    
    #Load model
    if opt.task == 'finetune':
        model = T5_model()
        
    if torch.cuda.device_count() > 1:
        print('You use %d GPUs'%torch.cuda.device_count())
        model = nn.DataParallel(model, device_ids=[0])
        torch.cuda.set_device(int(opt.cuda))
        model.cuda(int(opt.cuda))
        if opt.task == 'finetune':
            loss_fn = nn.CrossEntropyLoss(ignore_index = 0).cuda(int(opt.cuda))
        
    else:
        print('You use only one device')  
        device = torch.device("cuda" if USE_CUDA else "cpu")
        model.to(device)  
        if opt.task == 'finetune':
            loss_fn = nn.CrossEntropyLoss(ignore_index = 0)
        
    
    print('{:}s has {:} different named parameters.\n'.format(str(opt.model_name),len(list(model.named_parameters()))))
    
    #Prepare data
    df_train = pd.read_csv('data_train_test_val/train.csv')
    train_data = Dataload(df_train)
    train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size,shuffle=True,drop_last=False)
    
    df_val = pd.read_csv('data_train_test_val/val.csv')
    val_data = Dataload(df_val)
    val_loader = DataLoader(dataset=val_data, batch_size=opt.batch_size,shuffle=True,drop_last=False)
    
    
    tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
    
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=opt.learning_rate)
    
    # Total number of training steps is [number of batches] x [number of epochs]. 
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_data) * opt.epochs
    # Create the learning rate scheduler.
    #However scheduler is not needed for fine-tuning
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                num_warmup_steps = 0, # Default value in run_glue.py
                num_training_steps = total_steps)
    
    # We'll store a number of quantities such as training and validation loss etc
    training_stats = []
    
    #Path to save the best model
    best_loss = 10000
    early_stop = 0
  
    # For each epoch...
    for epoch_i in range(0, opt.epochs):
        total_time = time.time()
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, opt.epochs))
        print('Training...')
        
        model.train()
        
        # Reset the total loss for this epoch.
        tr_loss = 0.0
        
        
        for step, data in tqdm(enumerate(train_loader)):
            
            
            if torch.cuda.device_count() > 1:
                src_input_ids = data['src_input_ids'].cuda(non_blocking=True)
                src_attention_mask  = data['src_attention_mask'].cuda(non_blocking=True)
                lm_labels = data['tgt_input_ids'].cuda(non_blocking=True)
                lm_labels[lm_labels[:, :] == tokenizer.pad_token_id] = -100
                tgt_attention_mask = data['tgt_attention_mask'].cuda(non_blocking=True)
            else:    
                src_input_ids = data['src_input_ids'].to(device)
                src_attention_mask  = data['src_attention_mask'].to(device)
                lm_labels = data['tgt_input_ids'].to(device)
                lm_labels[lm_labels[:, :] == tokenizer.pad_token_id] = -100
                tgt_attention_mask = data['tgt_attention_mask'].to(device)
                
            # forward pass
            outputs = model(input_ids=src_input_ids, 
                        attention_mask=src_attention_mask,
                        labels=lm_labels,
                        decoder_attention_mask=tgt_attention_mask)
            
            loss = outputs[0]
            tr_loss += loss.item()
            
            model.zero_grad()
            loss.backward()
            del loss
            
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            
            
            if step%opt.statistic_step==0 and step!=0:
                print(f'-- Step: {step}')
                avg_train_loss = tr_loss / len(train_loader)
                print('Training loss:', avg_train_loss)
        
        print('{} seconds'.format(time.time() - total_time))   
        total_time = time.time() - total_time
        avg_train_loss = tr_loss / len(train_loader)
        print('Training loss:', avg_train_loss)
        
        print("")
        print("Running Validation...")
        
        model.eval()
        
        pred = []
        true = []
        
        cat_pred = []
        cat_true = []
        val_loss = 0.0
        
        with torch.no_grad():
        
            for step, data in tqdm(enumerate(val_loader)):
                
                if torch.cuda.device_count() > 1:
                    src_input_ids = data['src_input_ids'].cuda(non_blocking=True)
                    src_attention_mask  = data['src_attention_mask'].cuda(non_blocking=True)
                    tgt_input_ids = data['tgt_input_ids'].cuda(non_blocking=True)
                    lm_labels = tgt_input_ids.cuda(non_blocking=True)
                    lm_labels[lm_labels[:, :] == tokenizer.pad_token_id] = -100
                    tgt_attention_mask = data['tgt_attention_mask'].cuda(non_blocking=True)
                else:    
                    src_input_ids = data['src_input_ids'].to(device)
                    src_attention_mask  = data['src_attention_mask'].to(device)
                    tgt_input_ids = data['tgt_input_ids']
                    lm_labels = tgt_input_ids.to(device)
                    lm_labels[lm_labels[:, :] == tokenizer.pad_token_id] = -100
                    tgt_attention_mask = data['tgt_attention_mask'].to(device)
                    
                
                
                 
            
                # forward pass
                outputs = model(
                    input_ids=src_input_ids, 
                    attention_mask=src_attention_mask,
                    labels=lm_labels,
                    decoder_attention_mask=tgt_attention_mask)
                
                
                
                loss = outputs[0]
                
                
                
                val_loss += loss.item()
                
                # get true 
                for true_id in tgt_input_ids:
                    true_id[true_id[:] == -100] = 0
                    true.append(str(tokenizer.decode(true_id,skip_special_tokens=True)))
                    
                # get pred (decoder generated textual label ids)    
                pred_ids = model.model.generate(
                    input_ids=src_input_ids, 
                    attention_mask=src_attention_mask
                )
                
                
                pred_ids = pred_ids.cpu().numpy()
                for k,pred_id in enumerate(pred_ids):
                    pred_decoded = tokenizer.decode(pred_id,skip_special_tokens=True)
                    pred.append(str(pred_decoded))
                   
                if step%opt.statistic_step==0 and step!=0:   
                   loss_step = val_loss/(step+1)
                   print('Validation loss per %d steps:%0.3f'%(step,loss_step)) 
                      
                    
        #Convert labels to categorical
        for i in true:
            if i == label_names[0]:
                cat_true.append(0)
            else:    
                cat_true.append(1)
                        
        for j in pred:
            
            if j[:4]=='safe':
                cat_pred.append(0)
            elif j[:4]=='unsa':   
                cat_pred.append(1)
            else:
                cat_pred.append(2)   
                       
               
        # Report the final accuracy for this validation run.
        print("")
        print('Validation resuts')
        # Calculate the average loss over all of the batches.
        print(len(cat_true),len(cat_pred))
        avg_val_loss = val_loss / len(val_loader) 
        val_accuracy = accuracy_score(cat_true, cat_pred).item()
        print("Average val loss: {0:.2f}".format(avg_val_loss))
        print("Val laccuracy: {0:.2f}".format(val_accuracy))
        print(len(set(cat_pred)))
        if  2 in cat_pred:
            print(classification_report(cat_true, cat_pred, target_names=['Safe','Unsafe', 'Unknown'],zero_division=1))
        else:
            print(classification_report(cat_true, cat_pred, target_names=['Safe','Unsafe'],zero_division=1))
        print(pred[:20])
        df.loc[epoch_i] = pd.Series({'Epoch':int(epoch_i+1), 'Tr Loss':round(avg_train_loss,3), 'Val Loss':round(avg_val_loss,3), 'Time':round(total_time,2), 'lr':round(opt.learning_rate,6)})
                   
        if avg_val_loss<best_loss:
            early_stop =0
            best_loss = avg_val_loss
                
            print('Found better model')
            print("Saving model to %s" % os.path.join('best', '_'.join([str(i) for i in model_bin])))
            model_to_save = model
            torch.save(model_to_save, os.path.join('best', '_'.join([str(i) for i in model_bin])))
        else:
            early_stop +=1
            if (early_stop== opt.early_stop):
                print("Early stopping with best_val_loss: %.2f"%(best_loss))
                df.to_csv(os.path.join('best', '_'.join([str(i) for i in model_bin]) + '.csv'))
                break   
        
    #Save statistics to a txt file

    df.to_csv(os.path.join('best', '_'.join([str(i) for i in model_bin]) + '.csv'))
    print("")
    print("Training complete!")
            
    
    
if __name__ == "__main__":
    opt = parse_arguments()
    print(opt)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    USE_CUDA = torch.cuda.is_available()
    print("USE_CUDA")
    print(USE_CUDA)
    
    train(opt)

