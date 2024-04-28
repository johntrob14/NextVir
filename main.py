import torch
from schedulefree import AdamWScheduleFree
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModel
from utils import BinaryTrainer, Trainer, parse_multiclass_fa, TokenizedDataset, get_stack, get_criterion

def main(args):
    torch.manual_seed(args.seed)
    
    # Initialize Tokenizer and Base Model
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-S", trust_remote_code=True)
    model = AutoModel.from_pretrained("zhihan1996/DNABERT-S", trust_remote_code=True)
    
    # Parse fasta datasets
    data, bin_labels, (labels, conversion) = parse_multiclass_fa('./data/150bp_multiviral_train.fa')
    if args.num_classes == 1:
        training_dataset = TokenizedDataset(data, bin_labels, tokenizer, conversion=conversion)
    else:
        training_dataset = TokenizedDataset(data, labels, tokenizer, conversion=conversion)
    data, bin_labels, (labels, _) = parse_multiclass_fa('./data/150bp_multiviral_val.fa', class_names=conversion)
    if args.num_classes == 1:
        val_dataset = TokenizedDataset(data, bin_labels, tokenizer, conversion=conversion)
        # TODO (maybe): Let the Test function in the bin trainer handle multiclass labels for class specific outputs?
    else:
        val_dataset = TokenizedDataset(data, labels, tokenizer, conversion=conversion)
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
    # optimizer = AdamWScheduleFree(parameters, lr=lr)

    # Loaders and criterion
    train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    criterion = get_criterion(args, training_dataset)
    
    if len(device_list) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_list)
    
    # Train and Validate
    if args.num_classes == 1:
        trainer = BinaryTrainer(model, optimizer, criterion, device_list)
    else:
        trainer = Trainer(model, optimizer, criterion, device_list, args)
    
    for i in range(num_epochs):
        trainer.train(train_loader)
        trainer.validate(val_loader)
    
    # Test
    data, bin_labels, (labels, _) = parse_multiclass_fa('./data/150bp_multiviral_test.fa', class_names=conversion)
    if args.num_classes == 1:
        test_dataset = TokenizedDataset(data, bin_labels, tokenizer, conversion=conversion)
    else:
        test_dataset = TokenizedDataset(data, labels, tokenizer, conversion=conversion)
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
    opt.add_argument('--experiment', type=str)
    opt.add_argument('--seed', type=int, default=14)
    opt.add_argument('--debug', action='store_true')
    opt.add_argument('--num_classes', type=int, default=8)
    args=opt.parse_args()

    main(args)
    
    #TODO: add logging
    #TODO: move to DDP for multi-gpu training; should reduce overhead
    #TODO: Get someone to update the ROCM version (PLEASE!)
    #TODO: add support for other datasets and parsing individual fastas