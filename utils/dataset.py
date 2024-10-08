import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import pickle
from argparse import ArgumentParser
    
class TokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, tokenizer, conversion=None, args=None):
        self.conversion = conversion
        lens = []
        self.reads = [None] * len(data)
        self.token_type_ids = [None] * len(data)
        self.attention_mask = [None] * len(data)
        if args:
            match args.model:
                case 'dnabert-s':
                    max_len = 75
                case 'nt':
                    max_len = 150
                case 'hyenadna':
                    max_len = 151
                case _:
                    max_len = 75
        else:
            max_len = 75
        self.labels = torch.Tensor(labels)
        # Tokenize loop, change getitem
        batch_size = 32
        
        for i in range(0, len(data), batch_size):
            if i + batch_size > len(data):
                reads = data[i:]
            else:
                reads = data[i:i+batch_size]
            for j in range(len(reads)):
                if len(reads[j]) > max_len:
                    reads[j] = reads[j][:max_len]
            tokenized = tokenizer(reads, return_tensors = 'pt', padding='max_length', 
                                  max_length=max_len) # max_len=75 for D-S 150 for NT, 151 for H-DNA
                
            for k in range(len(reads)):
                # print(tokenized)
                self.reads[i + k] = tokenized['input_ids'][k]
                if 'token_type_ids' in tokenized:
                    self.token_type_ids[i + k] = tokenized['token_type_ids'][k]
                if 'attention_mask' in tokenized:
                    self.attention_mask[i + k] = tokenized['attention_mask'][k]
                    
        if 'token_type_ids' not in tokenized:
            self.token_type_ids = None
        if 'attention_mask' not in tokenized:
            self.attention_mask = None
                
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
        
    def remove_single(self, label):
        new_reads = []
        new_token_type_ids = []
        new_attention_mask = []
        new_labels = []
        num_removed = 0
        if label not in self.conversion:
            raise ValueError(f'Label not found, Labels: {self.conversion}, single_label: {label}')
        for i in range(len(self.reads)):
            if self.labels[i] != self.conversion.index(label):
                new_reads.append(self.reads[i])
                if self.attention_mask is not None:
                    new_attention_mask.append(self.attention_mask[i])
                if self.token_type_ids is not None:
                    new_token_type_ids.append(self.token_type_ids[i])
                new_labels.append(0.0 if self.labels[i] == 0 else 1.0)
            else:
                num_removed += 1
        print(num_removed)
        self.reads = torch.stack(new_reads)
        self.attention_mask = torch.stack(new_attention_mask) if self.attention_mask is not None else None
        self.token_type_ids = torch.stack(new_token_type_ids) if self.token_type_ids is not None else None
        self.labels = torch.Tensor(new_labels)
        
    def subsample_single(self, label):
        new_reads = []
        new_token_type_ids = []
        new_attention_mask = []
        new_labels = []
        if label not in self.conversion:
            raise ValueError(f'Label not found, Labels: {self.conversion}, single_label: {label}')
        for i in range(len(self.reads)):
            if self.labels[i] == self.conversion.index(label):
                new_reads.append(self.reads[i])
                if self.attention_mask is not None:
                    new_attention_mask.append(self.attention_mask[i])
                if self.token_type_ids is not None:
                    new_token_type_ids.append(self.token_type_ids[i])
                new_labels.append(1.0)
        vir_len = len(new_reads)
        hum_len = 0
        for i in range(len(self.reads)):
            if self.labels[i] == 0:
                new_reads.append(self.reads[i])
                if self.attention_mask is not None:
                    new_attention_mask.append(self.attention_mask[i])
                if self.token_type_ids is not None:
                    new_token_type_ids.append(self.token_type_ids[i])
                new_labels.append(0.0)
                hum_len += 1
                if hum_len >= vir_len:
                    break
        self.reads = torch.stack(new_reads)
        self.attention_mask = torch.stack(new_attention_mask) if self.attention_mask is not None else None
        self.token_type_ids = torch.stack(new_token_type_ids) if self.token_type_ids is not None else None
        self.labels = torch.Tensor(new_labels)
        
    def subsample_one_vs_all(self, label):
        for i in range(len(self.labels)):
            if self.labels[i].item() == self.conversion.index(label):
                self.labels[i] = 1
            else:
                self.labels[i] = 0
                
    def one_class_subsample(self, label):
        new_reads = []
        new_attention_mask = []
        new_token_type_ids = []
        for i in range(len(self.labels)):
            if self.labels[i] == self.conversion.index(label):
                new_reads.append(self.reads[i])
                if self.attention_mask is not None:
                    new_attention_mask.append(self.attention_mask[i])
                if self.token_type_ids is not None:
                    new_token_type_ids.append(self.token_type_ids[i])
        self.reads = torch.stack(new_reads)
        self.attention_mask = torch.stack(new_attention_mask) if self.attention_mask is not None else None
        self.token_type_ids = torch.stack(new_token_type_ids) if self.token_type_ids is not None else None
        self.labels = torch.ones(len(new_reads))

    def __len__(self):
        return len(self.reads)
    
    def __getitem__(self, index):
        output_dict = {'input_ids': self.reads[index]}
        if self.token_type_ids is not None:
            output_dict['token_type_ids'] = self.token_type_ids[index]
        if self.attention_mask is not None:
            output_dict['attention_mask'] = self.attention_mask[index]
        return output_dict, self.labels[index]
    

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

def parse_HPV_fa(filename):
    data = []
    labels = []
    with open(filename) as f:
        for line in f:
            if line.startswith('>'):
                if line.startswith('>HPV'):
                    labels.append(1)
                else:
                    labels.append(0)
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
            
def load_dataset(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
    
def parse_dataset(args, tokenizer, mode: str = 'train'):
    # Parse fasta datasets
    if mode == 'train':
        if args.train_path == 'XVir':
            conversion = ['HUM', 'HPV']
            data, bin_labels = parse_HPV_fa('./data/HPV/reads_150_train.fa')
        else:
            data, bin_labels, (labels, conversion) = parse_multiclass_fa(args.train_path)
        #TODO -- add support for other datasets, file paths
        dataset = None
        # if not args.test:
        if args.num_classes > 1 or args.single_label is not None:
            dataset = TokenizedDataset(data, labels, tokenizer, conversion=conversion)
            if args.single_label is not None:
                if args.one_vs_all:
                    dataset.subsample_one_vs_all(args.single_label)
                elif args.remove_single:
                    dataset.remove_single(args.single_label)
                else:
                    dataset.subsample_single(args.single_label)
        else:
            dataset = TokenizedDataset(data, bin_labels, tokenizer, conversion=conversion)
        print("training class spread: ", torch.unique(dataset.labels, return_counts=True))
        args.conversion = conversion
    elif mode == 'val':
        conversion = args.conversion
        
        if args.train_path == 'XVir':
            data, bin_labels = parse_HPV_fa('./data/HPV/reads_150_valid.fa')
        else:
            data, bin_labels, (labels, _) = parse_multiclass_fa(args.val_path, class_names=conversion)
        if args.num_classes > 1 or args.single_label is not None:
            dataset = TokenizedDataset(data, labels, tokenizer, conversion=conversion)
            if args.single_label is not None:
                if args.remove_single:
                    dataset.remove_single(args.single_label)
                elif args.one_vs_all:
                    dataset.subsample_one_vs_all(args.single_label)
                else:
                    dataset.subsample_single(args.single_label)
        else:
            dataset = TokenizedDataset(data, bin_labels, tokenizer, conversion=conversion)
    elif mode == 'test':
        conversion = args.conversion
        # this is where I'll have to add a bit of extra functionality
        if args.train_path == 'XVir':
            data, bin_labels = parse_HPV_fa('./data/HPV/reads_150_test.fa')
        else:
            data, bin_labels, (labels, _) = parse_multiclass_fa(args.test_path, class_names=conversion)

        if args.num_classes > 1 or args.single_label is not None:
            dataset = TokenizedDataset(data, labels, tokenizer, conversion=conversion)
            if args.single_label is not None:
                dataset.subsample_single(args.single_label)
        else:
            dataset = TokenizedDataset(data, bin_labels, tokenizer, conversion=conversion)
    return dataset
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, default='./data/')
    parser.add_argument('--filename', type=str, default='150bp_multiviral')
    parser.add_argument('--task', type=str, default='multi')
    parser.add_argument('--embedding', action='store_true')
    parser.add_argument('--tokenized', action='store_true')
    args = parser.parse_args()
    if args.embedding:
        save_embedding_dataset(args)
        
    print(load_dataset('./data/tokenized/150bp_multiviral_multi_train_tokenized_dataset.pkl'))