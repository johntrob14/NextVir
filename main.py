import torch
from schedulefree import AdamWScheduleFree
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModel
from utils import (Trainer, parse_multiclass_fa, parse_HPV_fa, TokenizedDataset, 
                   get_stack, get_criterion, parse_args, parse_dataset)
import wandb

def main(args):
    torch.manual_seed(args.seed)
    
    # Initialize Tokenizer and Base Model
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-S", trust_remote_code=True)
    model = AutoModel.from_pretrained("zhihan1996/DNABERT-S", trust_remote_code=True)        
    
    
    training_dataset = parse_dataset(args, tokenizer)
    
    val_dataset = parse_dataset(args, tokenizer, mode='val')
    torch.cuda.empty_cache()
    num_epochs = args.num_epochs
    lr = args.lr
    batch_size = args.batch_size
    device_list = list(map(int, args.device.split(',')))
    args.main_device = 'cuda:' + str(device_list[0]) # add to args for ease of use
    if args.verbose:
        print(device_list)
    
    model, parameters = get_stack(model, args)
    if len(device_list) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_list)
    if args.test:
        model.load_state_dict(torch.load(args.model_path))
        trainer = Trainer(model, None, None, device_list, args)
        test_dataset = parse_dataset(args, tokenizer, mode='test')
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        trainer.test(test_loader)
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
        
        # Test
        test_dataset = parse_dataset(args, tokenizer, mode='test')
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        trainer.load_best()
        trainer.test(test_loader)
    
if __name__ == '__main__':
    opt = ArgumentParser()
    args = parse_args(opt)
    main(args)
    
    
    #TODO: move to DDP for multi-gpu training; should reduce overhead
    #TODO: Get someone to update the ROCM version (PLEASE!)
    #TODO: add support for other datasets and parsing individual fastas
    #TODO: combine removal testing here
    #TODO: add mutation testing in dataset utils