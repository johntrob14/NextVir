import torch
from utils import set_lora, MultiClassifier, BinClassifier, AdapterStack
from argparse import ArgumentParser
import wandb

def get_stack(model, args):
    # Disable gradients in base model
    for param in model.parameters():
        param.requires_grad = False
    
    # Add LoRA matrices to Q,K,V and FF layers
    if not args.embedding_only:
        if args.verbose:
            print('Pre-LoRA:')  
            print(model)
            
        set_lora(model)
        if args.verbose:
            print('Post-LoRA:')
            print(model)   
     
        lora_params = [param for name, param in model.named_parameters() if 'lora' in name]                   
        if args.verbose:
            lora_names = [name for name, param in model.named_parameters() if 'lora' in name]
            print('LoRA Parameters:')
            print(lora_names)
    
    #Get adapter ready
    if args.num_classes == 1:
        adapter = BinClassifier(768)
        # adapter = BinClassifier(1024)
        # adapter = BinClassifier(256)
    else:
        adapter = MultiClassifier(768, num_classes=args.num_classes)
        # adapter = MultiClassifier(1024, num_classes=args.num_classes)
        # adapter = MultiClassifier(256, num_classes=args.num_classes)
    for param in adapter.parameters():
        param.requires_grad = True    

    # Configure Optimizer and stack the encoder with the adapter
    if not args.embedding_only:
        parameters = [{"params" : lora_params, "lr": args.lr},
                    {"params" : adapter.parameters(), "lr": args.lr}]
    else:
        parameters = [{"params" : adapter.parameters(), "lr": args.lr}]
        
    return AdapterStack(model, adapter).to(args.main_device), parameters

def get_criterion(args, training_dataset):
    if args.num_classes == 1:
        criterion = torch.nn.BCEWithLogitsLoss()
        if args.one_vs_all:
            totals = torch.unique(training_dataset.labels, return_counts=True)[1]
            sum = totals.sum()
            pos_weight = sum / (len(totals) * totals[1])
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        if args.verbose:
            print('Binary Classification')
    elif args.unweighted_loss:
        criterion = torch.nn.CrossEntropyLoss()
        if args.verbose:
            print('Unweighted Loss')
    else: 
        weights = torch.ones(len(training_dataset.conversion))
        totals = torch.unique(training_dataset.labels, return_counts=True)[1]
        sum = totals.sum()
        for i in range(len(totals)):
            weights[i] = sum / (len(totals) * totals[i])
        criterion = torch.nn.CrossEntropyLoss(weight=weights.to(args.main_device))
        if args.verbose:
            print('Loss Weights:')
            print(weights)
    return criterion

def parse_args(opt: ArgumentParser):
    opt.add_argument('--beta', type=float, default=0.85)
    opt.add_argument('--weight_decay', type=float, default=0.005)
    opt.add_argument('--batch_size', type=int, default=128)
    opt.add_argument('--lr', type=float, default=1e-3)
    opt.add_argument('--num_epochs', type=int, default=15)
    opt.add_argument('--embedding_only', action='store_true')
    opt.add_argument('--unweighted_loss', action='store_true')
    opt.add_argument('--save_path', type=str, default='./models')
    opt.add_argument('--device', type=str, default='4,5,6,7')
    opt.add_argument('--remove_single', action='store_true', help='Remove single class from train/val dataset')
    opt.add_argument('--verbose', type=bool, default=True) # always true right now
    opt.add_argument('--experiment', type=str) # will add logging to this subdirectory
    opt.add_argument('--seed', type=int, default=14)
    opt.add_argument('--debug', action='store_true')
    opt.add_argument('--tag', type=str, default=None)
    opt.add_argument('--num_classes', type=int, default=8,
                     help='Number of classes for classification - 1 for binary')
    opt.add_argument('--single_label', type=str, default=None,
                     help='Specify a single class for binary classification, ie "HHV-8" or "HTLV"')
    opt.add_argument('--one_vs_all', action='store_true')
    opt.add_argument('--test', action='store_true')
    opt.add_argument('-i', '--train_path', type=str, default='./data/150bp_multiviral_train.fa')
    opt.add_argument('-v', '--val_path', type=str, default='./data/150bp_multiviral_val.fa')
    opt.add_argument('-t', '--test_path', type=str, default='./data/150bp_multiviral_test.fa')
    opt.add_argument('-m', '--model_path', type=str, default='./models/best_model.pth')
    args=opt.parse_args()
    config = {
        'device': args.device,
        'beta': args.beta,
        'weight_decay': args.weight_decay,
        'lr': args.lr,
        'num_classes': args.num_classes,
        'single_label': args.single_label,
        'embedding_only': args.embedding_only,
    }
    if args.weight_decay == 0:
        config['weight_decay'] = 'None'
        args.weight_decay = int(0)
    if args.num_classes != 1 and args.single_label is not None:
        raise ValueError('Cannot specify single_label with multiclass classification')
    if args.tag is not None:
        wandb.init(project='NextVir', name=args.experiment, config=config, tags=[args.tag])
    else:
        wandb.init(project='NextVir', name=args.experiment, config=config)
    return args