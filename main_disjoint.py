import torch
from schedulefree import AdamWScheduleFree
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModel
from utils import Trainer, parse_multiclass_fa, parse_HPV_fa, TokenizedDataset, get_stack, get_criterion
import wandb

def main(args):
    torch.manual_seed(args.seed)
    
    # Initialize Tokenizer and Base Model
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-S", trust_remote_code=True)
    model = AutoModel.from_pretrained("zhihan1996/DNABERT-S", trust_remote_code=True)        
    
    # Parse fasta datasets
    data, bin_labels, (labels, conversion) = parse_multiclass_fa('./data/train_disjoint_fixed.fa')
    # conversion = ['HUM', 'HPV']
    # data, bin_labels = parse_HPV_fa('./data/HPV/reads_150_train.fa')
    if args.num_classes > 1 or args.single_label is not None:
        training_dataset = TokenizedDataset(data, labels, tokenizer, conversion=conversion)
        if args.single_label is not None and not args.one_vs_all:
            training_dataset.subsample_single(args.single_label)
        elif args.single_label is not None and args.one_vs_all:
            training_dataset.subsample_one_vs_all(args.single_label)
    else:
        training_dataset = TokenizedDataset(data, bin_labels, tokenizer, conversion=conversion)

    if args.verbose:
        print("training class spread: ", torch.unique(training_dataset.labels, return_counts=True))
    data, bin_labels, (labels, _) = parse_multiclass_fa('./data/val_disjoint_fixed.fa', class_names=conversion)
    # data, bin_labels = parse_HPV_fa('./data/HPV/reads_150_valid.fa')
    if args.num_classes > 1 or args.single_label is not None:
        val_dataset = TokenizedDataset(data, labels, tokenizer, conversion=conversion)
        if args.single_label is not None and not args.one_vs_all:
            val_dataset.subsample_single(args.single_label)
        elif args.single_label is not None and args.one_vs_all:
            val_dataset.subsample_one_vs_all(args.single_label)
    else:
        val_dataset = TokenizedDataset(data, bin_labels, tokenizer, conversion=conversion)

    torch.cuda.empty_cache()
    
    
    num_epochs = args.num_epochs
    lr = args.lr
    batch_size = args.batch_size
    device_list = list(map(int, args.device.split(',')))
    
    args.main_device = 'cuda:' + str(device_list[0]) # add to args for ease of use
    if args.verbose:
        print(device_list)
    
    model, parameters = get_stack(model, args)
    
    optimizer = AdamWScheduleFree(parameters, lr=lr, weight_decay=0.004)

    # Loaders and criterion
    train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    criterion = get_criterion(args, training_dataset)
    
    if len(device_list) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_list)
    
    # Train and Validate
    trainer = Trainer(model, optimizer, criterion, device_list, args)
    for i in range(num_epochs):
        trainer.train(train_loader)
        trainer.validate(val_loader)
    
    # Test
    data, bin_labels, (labels, _) = parse_multiclass_fa('./data/test_disjoint_fixed.fa', class_names=conversion)
    # data, bin_labels = parse_HPV_fa('./data/HPV/reads_150_test.fa')
    if args.num_classes > 1 or args.single_label is not None:
        test_dataset = TokenizedDataset(data, labels, tokenizer, conversion=conversion)
        if args.single_label is not None:
            test_dataset.subsample_single(args.single_label)
    else:
        test_dataset = TokenizedDataset(data, bin_labels, tokenizer, conversion=conversion)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    trainer.load_best()
    trainer.test(test_loader)
    
if __name__ == '__main__':
    opt = ArgumentParser()
    opt.add_argument('--batch_size', type=int, default=128)
    opt.add_argument('--lr', type=float, default=1e-3)
    opt.add_argument('--num_epochs', type=int, default=15)
    opt.add_argument('--embedding_only', action='store_true')
    opt.add_argument('--unweighted_loss', action='store_true')
    opt.add_argument('--save_path', type=str, default='./models')
    opt.add_argument('--device', type=str, default='4,5,6,7')
    opt.add_argument('--verbose', type=bool, default=True) # always true right now
    opt.add_argument('--experiment', type=str) # will add logging to this subdirectory
    opt.add_argument('--seed', type=int, default=14)
    opt.add_argument('--debug', action='store_true')
    opt.add_argument('--num_classes', type=int, default=8,
                     help='Number of classes for classification - 1 for binary')
    opt.add_argument('--single_label', type=str, default=None,
                     help='Specify a single class for binary classification, ie "HHV-8" or "HTLV"')
    opt.add_argument('--one_vs_all', action='store_true')
    args=opt.parse_args()
    if args.num_classes != 1 and args.single_label is not None:
        raise ValueError('Cannot specify single_label with multiclass classification')
    wandb.init(project='NextVir', name=args.experiment, config=args)
    main(args)
    
    
    #TODO: move to DDP for multi-gpu training; should reduce overhead
    #TODO: Get someone to update the ROCM version (PLEASE!)
    #TODO: add support for other datasets and parsing individual fastas