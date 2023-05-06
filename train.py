import os
import torch
import torch.nn as nn
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score
from sklearn.metrics import classification_report
from argument_parser import parse_arguments
from models.finetune import RobertaClass
from models.multitask import RoBERTa_MLM
from models.entailment import RoBERTa_Entailment
from transformers import get_linear_schedule_with_warmup
import pandas as pd
from data_loader import Dataload
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
import time
from random import shuffle
from models.mlm_prompt import RoBERTa_MLM_Prompt
# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


class features_MLM(nn.Module):
    def __init__(self):
        super(features_MLM, self).__init__()
        self.max_pred = 20
        
        self.tokenizer  = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
    def forward(self, files,labels):
        
        ids = []
        masks = []
        mlm_tokens = []
        mlm_pos = []
        labels_int = []
        
        for i,file in enumerate(files):
            
            with open(file, "r", errors='ignore') as f:
                  code_snippet = str(f.read())
            
            
            code = ''      
            for line in code_snippet.split('\n'):
                #disrecard language information, i.e., comments
                line = line.split('#')
                if len(line[0])>0 and '#' not in line[0]:
                    code += line[0] + ' '
                elif len(line)>1 and len(line[1])>0 and '#' not in line[1]: #When the comment followed by code
                    code += line[1] + ' '    
            
            if len(code)==0:
                continue #File is empty
                
            if labels[i] == 'safe':
                label = 0
            else:
                label = 1
            labels_int.append(label)
        
                  
            #tokens = self.tokenizer.tokenize(code)
            
            encoded_dict = self.tokenizer.encode_plus(
                code,  # document to encode.
                add_special_tokens=True,  # add '[CLS]' and '[SEP]'
                padding='max_length',  # set max length
                truncation=True,  # truncate longer messages
                pad_to_max_length=True,  # add padding
                return_attention_mask=True,  # create attn. masks
                return_tensors='pt'  # return pytorch tensors
            )
            
            input_ids = encoded_dict['input_ids'].squeeze(dim=0)
            attention_mask = encoded_dict['attention_mask'].squeeze(dim=0)
            
            
            
            n_pred =  min(self.max_pred, max(1, int(round(len(self.tokenizer.encode(code,add_special_tokens=False, truncation=True)) * 0.15)))) # 15 % of tokens in one sequence
            cand_maked_pos = [i for i, token in enumerate(self.tokenizer.encode(code,add_special_tokens=False,truncation=True)) ]
            shuffle(cand_maked_pos)
            
            masked_tokens, masked_pos = [], []
            for pos in cand_maked_pos[:n_pred]:
                masked_pos.append(pos)
                masked_tokens.append(input_ids[pos].item())
                input_ids[pos] = self.tokenizer.mask_token_id
            
            n_pad = self.max_pred - n_pred
            masked_tokens.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)
            
            
            ids.append(input_ids)
            masks.append(attention_mask)
            mlm_tokens.append(torch.tensor(masked_tokens))
            mlm_pos.append(torch.tensor(masked_pos))
            
        ids = torch.stack(ids, dim=0)
        masks = torch.stack(masks, dim=0)
        mlm_tokens = torch.stack(mlm_tokens, dim=0)
        mlm_pos = torch.stack(mlm_pos, dim=0)
        labels_int = torch.tensor(labels_int)
        return ids,masks,mlm_tokens,mlm_pos,labels_int      


class features_finetune(nn.Module):
    def __init__(self):
        super(features_finetune, self).__init__()
        
        self.tokenizer  = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    def forward(self, files,labels):
        
        ids = []
        masks = []
        labels_int = []
        for i,file in enumerate(files):
            
            with open(file, "r", errors='ignore') as f:
                  code_snippet = str(f.read())
            
            code = ''      
            for line in code_snippet.split('\n'):
                #disrecard language information, i.e., comments
                line = line.split('#')
                if len(line[0])>0 and '#' not in line[0]:
                    code += line[0] + ' '
                elif len(line)>1 and len(line[1])>0 and '#' not in line[1]: #When the comment followed by code
                    code += line[1] + ' '    
            
            if len(code)==0:
                continue
            
            if labels[i] == 'safe':
                label = 0
            else:
                label = 1
            labels_int.append(label)
            
            
            
            
            
            encoded_dict = self.tokenizer.encode_plus(
              code,                      # Sentence to encode.
              add_special_tokens=True,  # add '[CLS]' and '[SEP]'
              padding='max_length',  # set max length
              truncation=True,  # truncate longer messages
              pad_to_max_length=True,  # add padding
              return_attention_mask=True,  # create attn. masks
              return_tensors='pt'  # return pytorch tensors
              )  
           
           
            
            input_ids = encoded_dict['input_ids'].squeeze(dim=0)
            attention_mask = encoded_dict['attention_mask'].squeeze(dim=0)
            
            ids.append(input_ids)
            masks.append(attention_mask)
        
            
        ids = torch.stack(ids, dim=0)
        masks = torch.stack(masks, dim=0)
        labels_int = torch.tensor(labels_int)
        return ids,masks , labels_int
    
class features_Entail(nn.Module):
    def __init__(self):
        super(features_Entail, self).__init__()
        self.tokenizer  = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        self.dict = {
              "safe": "safe",  
              "unsafe": "unsafe"
            }
    def forward(self, files,labels):
        
        ids = []
        masks = []
        labels_int = []
        for i,file in enumerate(files):
            
            with open(file, "r", errors='ignore') as f:
                  code_snippet = str(f.read())
            
            code = ''      
            for line in code_snippet.split('\n'):
                #disrecard language information, i.e., comments
                line = line.split('#')
                if len(line[0])>0 and '#' not in line[0]:
                    code += line[0] + ' '
                elif len(line)>1 and len(line[1])>0 and '#' not in line[1]: #When the comment followed by code
                    code += line[1] + ' '    
                    
            if len(code)==0:
                continue
            
            en_ids = []
            en_masks = []
            for i in range (len(self.dict)):
                code_des = list(self.dict.values())[i] + '.' +  code
                
            
                encoded_dict = self.tokenizer.encode_plus(
                    code_des,  # document to encode.
                    add_special_tokens=True,  # add '[CLS]' and '[SEP]'
                    padding='max_length',  # set max length
                    truncation=True,  # truncate longer messages
                    pad_to_max_length=True,  # add padding
                    return_attention_mask=True,  # create attn. masks
                    return_tensors='pt'  # return pytorch tensors
                )
                
                input_ids = encoded_dict['input_ids'].squeeze(dim=0)
                attention_mask = encoded_dict['attention_mask'].squeeze(dim=0)
                
                en_ids.append(input_ids)
                en_masks.append(attention_mask)
            
            en_ids = torch.stack(en_ids, dim=0)
            en_masks = torch.stack(en_masks, dim=0)
            ids.append(en_ids)
            masks.append(en_masks)
            
            
            if labels[i] == 'safe':
                label = 0
            else:
                label = 1
            labels_int.append(label)
            
        ids = torch.stack(ids, dim=0)
        masks = torch.stack(masks, dim=0)
        labels_int = torch.tensor(labels_int)
        return ids, masks, labels_int


class features_prompt(nn.Module):
    def __init__(self):
        super(features_prompt, self).__init__()
        
        self.tokenizer  = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
        self.special_tokens_dict = {'additional_special_tokens': ['safe','unsafe', '<sep>']}
        self.num_added_toks = self.tokenizer.add_special_tokens(self.special_tokens_dict)
        #self.tokenizer.all_special_tokens
        #self.tokenizer.all_special_ids
        
    def forward(self, files,labels):
        
        ids = []
        masks = []
        mlm_tokens = []
        mlm_pos = []
        labels_int = []
        for i,file in enumerate(files):
            
            with open(file, "r", errors='ignore') as f:
                  code_snippet = str(f.read())
            
            code = ''      
            for line in code_snippet.split('\n'):
                #disrecard language information, i.e., comments
                line = line.split('#')
                if len(line[0])>0 and '#' not in line[0]:
                    code += line[0] + ' '
                elif len(line)>1 and len(line[1])>0 and '#' not in line[1]: #When the comment followed by code
                    code += line[1] + ' ' 
                    
            if len(code)==0:
                continue
            
            if labels[i] == 'safe':
                label = 0
            else:
                label = 1
            labels_int.append(label)
            
            code_tokens = self.tokenizer.tokenize(code)
            prompt_tokens = self.tokenizer.tokenize('<sep> It is <mask>')
            
            if len(code_tokens)> 506:
                code_tokens = code_tokens[:506]
            
            tokens = code_tokens + prompt_tokens    
            
            
            encoded_dict = self.tokenizer.encode_plus(
                tokens,  # document to encode.
                add_special_tokens=True,  # add '[CLS]' and '[SEP]'
                padding='max_length',  # set max length
                truncation=True,  # truncate longer messages
                pad_to_max_length=True,  # add padding
                return_attention_mask=True,  # create attn. masks
                return_tensors='pt'  # return pytorch tensors
            )
            
            input_ids = encoded_dict['input_ids'].squeeze(dim=0)
            attention_mask = encoded_dict['attention_mask'].squeeze(dim=0)
            
            masked_tokens, masked_pos = [], []
            masked_tokens.append(self.tokenizer.encode(self.tokenizer.tokenize(labels[i]),add_special_tokens=False)[0])
            masked_pos.append((input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0])
               
                
            ids.append(input_ids)
            masks.append(attention_mask)
            mlm_tokens.append(torch.tensor(masked_tokens))
            mlm_pos.append(torch.tensor(masked_pos))
            
        ids = torch.stack(ids, dim=0)
        masks = torch.stack(masks, dim=0)
        mlm_tokens = torch.stack(mlm_tokens, dim=0)
        mlm_pos = torch.stack(mlm_pos, dim=0)
        labels_int = torch.tensor(labels_int)
        return ids,masks,mlm_tokens,mlm_pos,labels_int 

def train(opt):
    label_names = ["safe","unsafe"]
    
    df = pd.DataFrame(columns = ['Epoch','Tr Loss', 'Val Loss', 'Tr Acc', 'Val Acc',  'Time','lr'])
    model_bin = [opt.task, opt.model_name, opt.batch_size, opt.learning_rate]
   
    
    #Load model
    if opt.task == 'finetune':
        model = RobertaClass()
    elif opt.task == 'multitask':
        model = RoBERTa_MLM()
    elif opt.task == 'prompt':
        model = RoBERTa_MLM_Prompt()
    
    if torch.cuda.device_count() > 1:
        print('You use %d GPUs'%torch.cuda.device_count())
        model = nn.DataParallel(model, device_ids=[0,1,2,3])
        torch.cuda.set_device(int(opt.cuda))
        model.cuda(int(opt.cuda))
        if opt.task == 'finetune' or opt.task == 'prompt':
            loss_fn = nn.CrossEntropyLoss().cuda(int(opt.cuda))
        elif opt.task == 'multitask':
            loss_ml = nn.CrossEntropyLoss(ignore_index = 0).cuda(int(opt.cuda))
            loss_clf = nn.CrossEntropyLoss().cuda(int(opt.cuda))
            
        
    else:
        print('You use only one device')  
        device = torch.device("cuda" if USE_CUDA else "cpu")
        model.to(device)  
        if opt.task == 'finetune' or opt.task == 'prompt':
            loss_fn = nn.CrossEntropyLoss()
        elif opt.task == 'multitask':
            loss_ml = nn.CrossEntropyLoss(ignore_index = 0)
            loss_clf = nn.CrossEntropyLoss()
    
    print('{:}s has {:} different named parameters.\n'.format(str(opt.model_name),len(list(model.named_parameters()))))

    
    #Prepare data
    df_train = pd.read_csv('data_train_test_val/train.csv')
    train_data = Dataload(df_train)
    train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size,shuffle=True,drop_last=False)
    
    df_val = pd.read_csv('data_train_test_val/val.csv')
    val_data = Dataload(df_val)
    val_loader = DataLoader(dataset=val_data, batch_size=opt.batch_size,shuffle=True,drop_last=False)
    
    if  opt.task == 'finetune':
        features = features_finetune()
    elif opt.task == 'multitask':   
        features = features_MLM()
    elif opt.task == 'prompt':
        features =  features_prompt()   
        
        
    if opt.task == 'prompt':
        tokenizer  = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
        special_tokens_dict = {'additional_special_tokens': ['safe','unsafe', '<sep>']}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
       
    
        
    
    optimizer = torch.optim.Adam(params=model.parameters(), lr=opt.learning_rate)
    
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
        n_predict = []
        n_labels = []
        
        for step, (files, labels) in tqdm(enumerate(train_loader)):
            
           if opt.task == 'finetune':
                _input, _mask, _labels = features(files,labels)
                
                #Load data
                if torch.cuda.device_count() > 1:
                    _input = _input.cuda(non_blocking=True)
                    _mask  = _mask.cuda(non_blocking=True)
                    _labels = _labels.cuda(non_blocking=True)
                else:    
                    _input = _input.to(device)
                    _mask  = _mask.to(device)
                    _labels = _labels.to(device)
                    
                #Calculate loss
                outputs = model(_input, _mask)
                loss = loss_fn(outputs,_labels)
                    
           elif opt.task == 'multitask':   
               
                _input,_mask, _mlm_tokens,_mlm_pos, _labels = features(files,labels)
                #Load data
                if torch.cuda.device_count() > 1:
                    _input = _input.cuda(non_blocking=True)
                    _mask  = _mask.cuda(non_blocking=True)
                    _mlm_tokens = _mlm_tokens.cuda(non_blocking=True)
                    _mlm_pos = _mlm_pos.cuda(non_blocking=True)
                    _labels = _labels.cuda(non_blocking=True)
                else:    
                    _input = _input.to(device)
                    _mask  = _mask.to(device)
                    _mlm_tokens  = _mlm_tokens.to(device)
                    _mlm_pos  = _mlm_pos.to(device)
                    _labels = _labels.to(device)
                
                outputs = model(_input, _mask, _mlm_pos, _mlm_pos)  
                
                loss_1 = loss_clf(outputs[0],_labels)
                loss_2 = loss_ml(outputs[1].view(-1, 50265),_mlm_tokens.view(-1))
                
                loss = (1/0.115)*loss_1 + (1/5.695)*loss_2
           elif opt.task == 'prompt':  
               _input,_mask, _mlm_tokens,_mlm_pos, _labels = features(files,labels)
               
               if torch.cuda.device_count() > 1:
                   _input = _input.cuda(non_blocking=True)
                   _mask  = _mask.cuda(non_blocking=True)
                   _mlm_tokens = _mlm_tokens.cuda(non_blocking=True)
                   _mlm_pos = _mlm_pos.cuda(non_blocking=True)
                   _labels = _labels.cuda(non_blocking=True)
               else:    
                   _input = _input.to(device)
                   _mask  = _mask.to(device)
                   _mlm_tokens  = _mlm_tokens.to(device)
                   _mlm_pos  = _mlm_pos.to(device)
                   _labels = _labels.to(device)
                
               output = model(_input, _mask, _mlm_tokens, _mlm_pos) 
               
               loss = loss_fn(output.view(-1, 50269),_mlm_tokens.view(-1))
               
               
           tr_loss += loss.item()
          
           model.zero_grad()
           loss.backward()
           del loss
           
           torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
           optimizer.step()
           scheduler.step()
           
           if opt.task == 'finetune' :
               n_predict.extend(np.argmax(outputs.cpu().detach().numpy(),axis=-1))
           elif opt.task == 'multitask':
               n_predict.extend(np.argmax(outputs[0].cpu().detach().numpy(),axis=-1))
           elif opt.task == 'prompt':
               maxs = torch.argmax(output.cpu().detach(), dim=-1)
               predictions = []
               for _max in maxs:
                   predictions.append(tokenizer.decode(_max))
                   
               for pred in predictions:
                   if pred == label_names[0]:
                       n_predict.append(0)
                   elif pred == label_names[1]:   
                       n_predict.append(1)
                   else:
                       n_predict.append(2)    
               
           n_labels.extend(_labels.cpu().detach().numpy())
           
            
           if step%opt.statistic_step==0 and step!=0:
                loss_step = tr_loss/(step+1)
                train_accuracy = accuracy_score(n_labels, n_predict).item()
                print('Train loss per %d steps:%0.3f'%(step,loss_step))
                if len(set(n_predict))==3:
                    print(classification_report(n_labels, n_predict, target_names=['Safe','Unsafe', 'Unknown'],zero_division=1))
                else:
                    print(classification_report(n_labels, n_predict, target_names=['Safe','Unsafe']))
                
                    
                 
        print('{} seconds'.format(time.time() - total_time))   
        total_time = time.time() - total_time
        print("")
        print('Training resuts')    
        # Calculate the average loss over all of the batches.
        avg_train_loss = tr_loss / len(train_loader) 
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        if len(set(n_predict))==3:
            print(classification_report(n_labels, n_predict, target_names=['Safe','Unsafe', 'Unknown'],zero_division=1))
        else:
            print(classification_report(n_labels, n_predict, target_names=['Safe','Unsafe']))
                 
        
        
        print("")
        print("Running Validation...")
        
        model.eval()
        
        n_predict = []
        n_labels = []
        val_loss = 0
        with torch.no_grad():   
            # Evaluate data for one epoch
            for step, (files, labels) in tqdm(enumerate(train_loader)):
               if opt.task == 'finetune' or opt.task == 'entailment':
                    _input, _mask, _labels = features(files,labels)
                    
                    #Load data
                    if torch.cuda.device_count() > 1:
                        _input = _input.cuda(non_blocking=True)
                        _mask  = _mask.cuda(non_blocking=True)
                        _labels = _labels.cuda(non_blocking=True)
                    else:    
                        _input = _input.to(device)
                        _mask  = _mask.to(device)
                        _labels = _labels.to(device)
                        
                    #Calculate loss
                    outputs = model(_input, _mask)
                    loss = loss_fn(outputs,_labels)
                        
               elif opt.task == 'multitask':   
                   
                    _input,_mask, _mlm_tokens,_mlm_pos, _labels = features(files,labels)
                    #Load data
                    if torch.cuda.device_count() > 1:
                        _input = _input.cuda(non_blocking=True)
                        _mask  = _mask.cuda(non_blocking=True)
                        _mlm_tokens = _mlm_tokens.cuda(non_blocking=True)
                        _mlm_pos = _mlm_pos.cuda(non_blocking=True)
                        _labels = _labels.cuda(non_blocking=True)
                    else:    
                        _input = _input.to(device)
                        _mask  = _mask.to(device)
                        _mlm_tokens  = _mlm_tokens.to(device)
                        _mlm_pos  = _mlm_pos.to(device)
                        _labels = _labels.to(device)
                    
                    outputs = model(_input, _mask, _mlm_pos, _mlm_pos)  
                    
                    loss_1 = loss_clf(outputs[0],_labels)
                    loss_2 = loss_ml(outputs[1].view(-1, 50265),_mlm_tokens.view(-1))
                    
                    loss = (1/0.115)*loss_1 + (1/5.695)*loss_2

               elif opt.task == 'prompt':  
                   _input,_mask, _mlm_tokens,_mlm_pos, _labels = features(files,labels)
                   
                   if torch.cuda.device_count() > 1:
                       _input = _input.cuda(non_blocking=True)
                       _mask  = _mask.cuda(non_blocking=True)
                       _mlm_tokens = _mlm_tokens.cuda(non_blocking=True)
                       _mlm_pos = _mlm_pos.cuda(non_blocking=True)
                       _labels = _labels.cuda(non_blocking=True)
                   else:    
                       _input = _input.to(device)
                       _mask  = _mask.to(device)
                       _mlm_tokens  = _mlm_tokens.to(device)
                       _mlm_pos  = _mlm_pos.to(device)
                       _labels = _labels.to(device)
                    
                   output = model(_input, _mask, _mlm_tokens, _mlm_pos) 
                   
                   loss = loss_fn(output.view(-1, 50269),_mlm_tokens.view(-1))
                   
               # Accumulate the validation loss.
               val_loss += loss.item()  
               
               
               #Precision, Recall, F1
               if opt.task == 'finetune' :
                   n_predict.extend(np.argmax(outputs.cpu().detach().numpy(),axis=-1))
               elif opt.task == 'multitask':
                   n_predict.extend(np.argmax(outputs[0].cpu().detach().numpy(),axis=-1))
               elif opt.task == 'prompt':
                   maxs = torch.argmax(output.cpu().detach(), dim=-1)
                   predictions = []
                   for _max in maxs:
                       predictions.append(tokenizer.decode(_max))
                       
                   for pred in predictions:
                       if pred == label_names[0]:
                           n_predict.append(0)
                       elif pred == label_names[1]:   
                           n_predict.append(1)
                       else:
                           n_predict.append(2)    
                   
               n_labels.extend(_labels.cpu().detach().numpy())
               
                
               if step%opt.statistic_step==0 and step!=0:
                    loss_step = val_loss/(step+1)
                    val_accuracy = accuracy_score(n_labels, n_predict).item()
                    print('Validation loss per %d steps:%0.3f'%(step,loss_step))
                    if len(set(n_predict))==3:
                        print(classification_report(n_labels, n_predict, target_names=['Safe','Unsafe', 'Unknown'],zero_division=1))
                    else:
                        print(classification_report(n_labels, n_predict, target_names=['Safe','Unsafe']))
                    
                        
        # Report the final accuracy for this validation run.
        print("")
        print('Validation resuts')
        # Calculate the average loss over all of the batches.
        avg_val_loss = val_loss / len(val_loader) 
        print("Average val loss: {0:.2f}".format(avg_val_loss))
        if len(set(n_predict))==3:
            print(classification_report(n_labels, n_predict, target_names=['Safe','Unsafe', 'Unknown'],zero_division=1))
        else:
            print(classification_report(n_labels, n_predict, target_names=['Safe','Unsafe']))
        
        df.loc[epoch_i] = pd.Series({'Epoch':int(epoch_i+1), 'Tr Loss':round(avg_train_loss,3), 'Val Loss':round(avg_val_loss,3), 'Tr Acc':round(train_accuracy,2), 'Val Acc':round(val_accuracy,2), 'Time':round(total_time,2), 'lr':round(opt.learning_rate,6)})
        
        
        
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
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    USE_CUDA = torch.cuda.is_available()
    print("USE_CUDA")
    print(USE_CUDA)
    #device = torch.device("cuda" if USE_CUDA else "cpu")
    
    train(opt)
