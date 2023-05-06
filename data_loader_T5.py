from torch.utils.data import Dataset
from transformers import RobertaTokenizer

class Dataload(Dataset):
    def __init__(self, df):
        super(Dataload, self).__init__()
        self.df = df
        self.file = df['File'].values
        self.category = df['Category'].values
        
        self.tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
        
        self.src_max_length = 512
        self.tgt_max_length = 2
        
    def __getitem__(self, index):
        
        file = self.file[index]
        
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
        label = self.category[index] 
        
        
        src_tokenized =  self.tokenizer.encode_plus(
            code,  # document to encode.
            add_special_tokens=True,  # add '[CLS]' and '[SEP]'
            max_length=self.src_max_length,
            padding='max_length',  # set max length
            truncation=True,  # truncate longer messages
            pad_to_max_length=True,  # add padding
            return_attention_mask=True,  # create attn. masks
            return_tensors='pt'  # return pytorch tensors
        )
        
        src_input_ids = src_tokenized['input_ids'].squeeze()
        src_attention_mask = src_tokenized['attention_mask'].squeeze()
        
        tgt_tokenized = self.tokenizer.encode_plus(
            label,  # document to encode.
            add_special_tokens=False,  # add '[CLS]' and '[SEP]'
            max_length=self.tgt_max_length,
            padding='max_length',  # set max length
            truncation=True,  # truncate longer messages
            pad_to_max_length=True,  # add padding
            return_attention_mask=True,  # create attn. masks
            return_tensors='pt'  # return pytorch tensors
        )
        
        tgt_input_ids = tgt_tokenized['input_ids'].squeeze()
        tgt_attention_mask = tgt_tokenized['attention_mask'].squeeze()
        
        return {
                'src_input_ids': src_input_ids.long(),
                'src_attention_mask': src_attention_mask.long(),
                'tgt_input_ids': tgt_input_ids.long(),
                'tgt_attention_mask': tgt_attention_mask.long()
            }

        
        
    def __len__ (self):
        return len(self.df)  