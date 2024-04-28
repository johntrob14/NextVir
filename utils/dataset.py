import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import pickle
from argparse import ArgumentParser

class EmbedDataset(torch.utils.data.Dataset):
    
    def __init__(self, data=None, labels=None, tokenizer=None, model=None, filename=None, conversion=None):
        if data is None or labels is None or tokenizer is None or model is None:
            raise ValueError('Must provide either data, labels, tokenizer, and model or a filename')
        lens = []
        model.to('cuda:3')
        # self.reads = torch.zeros(len(data), 768*363)
        self.reads = torch.zeros(len(data), 768)
        
        self.conversion = conversion
        dvf_max_len = 363
        self.labels = torch.Tensor(labels)
        # Tokenize loop, change getitem
        batch_size = 128
        for i in range(0, len(data), batch_size):
            if i + batch_size > len(data):
                reads = data[i:]
            else:
                reads = data[i:i+batch_size]
            tokenized = tokenizer(reads, return_tensors = 'pt', padding='longest', max_length=dvf_max_len)['input_ids']
            lens.append(tokenized.shape[1])
            tokenized = tokenized.to('cuda:3')
            embedded = model(tokenized)[0]
            # pooled = torch.flatten(embedded, start_dim=1).cpu().detach()
            pooled = embedded.mean(dim=1).cpu().detach()
            for k in range(len(reads)):
                self.reads[i + k] = pooled[k]
            del embedded, tokenized
    
        
    def subsample_data(self, max_samples):
        per_class_lengths = [0] * len(self.conversion)
        new_reads = []
        new_labels = []
        for i in range(len(self.reads)):
            if per_class_lengths[int(self.labels[i].argmax())] < max_samples:
                per_class_lengths[int(self.labels[i].argmax())] += 1
                new_reads.append(self.reads[i])
                new_labels.append(self.labels[i])
        self.reads = torch.stack(new_reads)
        self.labels = torch.stack(new_labels)
        print(per_class_lengths)
                
            
        
        
    def __len__(self):
        return len(self.reads)
    
    def __getitem__(self, index):
        return {'embedding': self.reads[index], 'labels' : self.labels[index]}
    
class TokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, tokenizer, conversion=None):
        self.conversion = conversion
        lens = []
        self.reads = [None] * len(data)
        self.token_type_ids = [None] * len(data)
        self.attention_mask = [None] * len(data)
        dvf_max_len = 363
        self.labels = torch.Tensor(labels)
        # Tokenize loop, change getitem
        batch_size = 32
        for i in range(0, len(data), batch_size):
            if i + batch_size > len(data):
                reads = data[i:]
            else:
                reads = data[i:i+batch_size]
            tokenized = tokenizer(reads, return_tensors = 'pt', padding='max_length', max_length=75)
            lens.append(tokenized['input_ids'].shape[1])
            for k in range(len(reads)):
                self.reads[i + k] = tokenized['input_ids'][k]
                self.token_type_ids[i + k] = tokenized['token_type_ids'][k]
                self.attention_mask[i + k] = tokenized['attention_mask'][k]
                
    def subsample_data(self, max_samples):
        per_class_lengths = [0] * len(self.conversion)
        new_reads = []
        new_labels = []
        for i in range(len(self.reads)):
            if per_class_lengths[int(self.labels[i].argmax())] < max_samples:
                per_class_lengths[int(self.labels[i].argmax())] += 1
                new_reads.append(self.reads[i])
                new_labels.append(self.labels[i])
        self.reads = torch.stack(new_reads)
        self.labels = torch.stack(new_labels)
        print(per_class_lengths)

    def __len__(self):
        return len(self.reads)
    
    def __getitem__(self, index):
        return {'input_ids': self.reads[index], 
                'attention_mask': self.attention_mask[index],
                'token_type_ids': self.token_type_ids[index]}, self.labels[index]
    

def parse_fa(filename):
    data = []
    labels = []
    with open(filename) as f:
        for line in f:
            if line.startswith('>'):
                labels.append(line.strip())
            else:
                data.append(line.strip())
    return data, labels

def parse_multiclass_fa(filename, class_names=['HUM']):
    data = []
    labels = []
    multiclass_labels = []
    with open(filename) as f:
        for line in f:
            if line.startswith('>'):
                if line.startswith('>VIR'):
                    labels.append(1)
                    vir_name = line.split('_')[1]
                    if vir_name not in class_names:
                        class_names.append(vir_name)
                    multiclass_labels.append(class_names.index(vir_name))
                else:
                    labels.append(0)
                    multiclass_labels.append(0)
            else:
                data.append(line.strip())
    return data, labels, (multiclass_labels, class_names)

def save_embedding_dataset(args):
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-S", trust_remote_code=True)
    model = AutoModel.from_pretrained("zhihan1996/DNABERT-S", trust_remote_code=True)
    class_names = ['HUM']
    for version in ['_train.fa', '_val.fa', '_test.fa']:
        if args.task == 'binary':
            data, labels, (_, _) = parse_multiclass_fa(args.filename + version)
            dataset = EmbedDataset(data, labels, tokenizer, model)
        elif args.task == 'multi':
            data, labels, (multiclass_labels, class_names) = parse_multiclass_fa(args.filename + version, class_names=class_names)
            dataset = EmbedDataset(data, multiclass_labels, tokenizer, model, conversion=class_names)
        with open(args.filename + '_' + args.task + version.replace('.fa','') + '_embed_dataset.pkl', 'wb') as f:
            pickle.dump(dataset, f)
            
def save_tokenized_dataset(args):
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-S", trust_remote_code=True)
    class_names = ['HUM']
    for version in ['_train.fa', '_val.fa', '_test.fa']:
        if args.task == 'binary':
            data, labels, (_, _) = parse_multiclass_fa(args.filename + version)
            dataset = TokenizedDataset(data, labels, tokenizer)
        elif args.task == 'multi':
            data, labels, (multiclass_labels, class_names) = parse_multiclass_fa(args.filename + version, class_names=class_names)
            dataset = TokenizedDataset(data, multiclass_labels, tokenizer, conversion=class_names)
        with open(args.filename + '_' + args.task + version.replace('.fa','') + '_tokenized_dataset.pkl', 'wb') as f:
            pickle.dump(dataset, f)
            
def load_dataset(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, default='./data/')
    parser.add_argument('--filename', type=str, default='150bp_multiviral')
    parser.add_argument('--task', type=str, default='multi')
    parser.add_argument('--embedding', action='store_true')
    parser.add_argument('--tokenized', action='store_true')
    args = parser.parse_args()
    if args.tokenized:
        save_tokenized_dataset(args)
    if args.embedding:
        save_embedding_dataset(args)
        
    print(load_dataset('./data/tokenized/150bp_multiviral_multi_train_tokenized_dataset.pkl'))