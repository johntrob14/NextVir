import torch
from utils import set_lora, MultiClassifier, BinClassifier, AdapterStack


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
    else:
        adapter = MultiClassifier(768, num_classes=args.num_classes)
    for param in adapter.parameters():
        param.requires_grad = True    

    # Configure Optimizer and stack the encoder with the adapter
    if not args.embedding_only:
        parameters = [{"params" : lora_params, "lr": args.lr/10, 'warmup_steps': 1000},
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