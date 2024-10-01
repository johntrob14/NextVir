import torch
from schedulefree import AdamWScheduleFree
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoModelForSequenceClassification
from utils import (Trainer, parse_multiclass_fa, parse_HPV_fa, TokenizedDataset, 
                   get_stack, get_criterion, parse_args, parse_dataset)
import wandb
from sklearn.metrics import roc_curve
import numpy as np

def main(args):
    torch.manual_seed(args.seed)
    
    # Initialize Tokenizer and Base Model
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-S", trust_remote_code=True)
    model = AutoModel.from_pretrained("zhihan1996/DNABERT-S", trust_remote_code=True)     
    # tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", trust_remote_code=True)
    # model = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", trust_remote_code=True)
    # tokenizer = AutoTokenizer.from_pretrained('LongSafari/hyenadna-large-1m-seqlen-hf', trust_remote_code=True)
    # model = AutoModelForSequenceClassification.from_pretrained('LongSafari/hyenadna-large-1m-seqlen-hf', trust_remote_code=True)
    
    training_dataset = parse_dataset(args, tokenizer)
    
    val_dataset = parse_dataset(args, tokenizer, mode='val')
    torch.cuda.empty_cache()
    num_epochs = args.num_epochs
    lr = args.lr
    batch_size = args.batch_size
    device_list = list(map(int, args.device.split(',')))
    args.main_device = 'cuda:' + str(device_list[0]) # add to args for ease of use
    # args.main_device = 'cpu'
    if args.verbose:
        print(device_list)
    
    model, parameters = get_stack(model, args)
        

    if len(device_list) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_list)
    if args.test:
        # model.load_state_dict(torch.load(args.model_path, map_location=device_list))
        # print('Loaded model from: ', args.model_path)
        print('LOADING')
        # print(torch.load(args.model_path))
        if len(device_list) > 1:
            model.module.load_state_dict(torch.load(args.model_path))
        else:
            model.load_state_dict(torch.load(args.model_path))
        print('Loaded model from: ', args.model_path)
        trainer = Trainer(model, None, None, device_list, args)
        test_dataset = parse_dataset(args, tokenizer, mode='test')
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        if args.num_classes == 1:
            (_, probs) = trainer.test(test_loader)
            probs = [str(p) + '\n' for p in probs]
            with open(args.single_label + '_results.csv', 'w') as f:
                f.writelines(probs)
        else:
            trainer.test(test_loader)
        # y_true, probs = trainer.test(test_loader)
        # fpr, tpr, thresh = roc_curve(y_true, probs)
        # np.save('NextVir_disjoint_tpr.npy', tpr)
        # np.save('NextVir_disjoint_fpr.npy', fpr)
        # np.save('NextVir_disjoint_thresh.npy', thresh)
    else:
        warmup_step_rate = 0.4
        warmup_steps = int(len(training_dataset) // batch_size * warmup_step_rate)
        print('warmup_steps: ', warmup_steps)
        optimizer = AdamWScheduleFree(parameters, lr=lr, warmup_steps=warmup_steps, 
                                    weight_decay=args.weight_decay,
                                    betas=(args.beta, 0.999))

        # Loaders and criterion
        train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        criterion = get_criterion(args, training_dataset)
        
        
        
        # Train and Validate
        trainer = Trainer(model, optimizer, criterion, device_list, args)
        for i in range(num_epochs):
            trainer.train(train_loader)
            trainer.validate(val_loader)
            if trainer.epochs_since_best >= 4:
                break
        
        # Test
        test_dataset = parse_dataset(args, tokenizer, mode='test')
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        trainer.load_best()
        # y_true, probs = trainer.test(test_loader)
        if args.num_classes == 1:
            (_, probs) = trainer.test(test_loader)
            probs = [str(p) + '\n' for p in probs]
            with open(args.single_label + '_results.csv', 'w') as f:
                f.writelines(probs)
        else:
            trainer.test(test_loader)
        # fpr, tpr, thresh = roc_curve(y_true, probs)
        # np.save('NextVir_disjoint_tpr.npy', tpr)
        # np.save('NextVir_disjoint_fpr.npy', fpr)
        # np.save('NextVir_disjoint_thresh.npy', thresh)
    
if __name__ == '__main__':
    opt = ArgumentParser()
    args = parse_args(opt)
    main(args)
    
    
    
    #TODO: add support for other datasets and parsing individual fastas
    #TODO: incorportate dataset mutation to dataset utils