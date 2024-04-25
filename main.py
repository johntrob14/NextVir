import torch

from schedulefree import AdamWScheduleFree
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModel
from utils import set_lora, Trainer, parse_multiclass_fa, TokenizedDataset, MultiClassifier, AdapterStack

def main(args):
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-S", trust_remote_code=True)
    model = AutoModel.from_pretrained("zhihan1996/DNABERT-S", trust_remote_code=True)
    data, _, (labels, conversion) = parse_multiclass_fa('./data/150bp_multiviral_train.fa')
    training_dataset = TokenizedDataset(data, labels, tokenizer, conversion=conversion)
    data, _, (labels, _) = parse_multiclass_fa('./data/150bp_multiviral_val.fa', class_names=conversion)
    val_dataset = TokenizedDataset(data, labels, tokenizer, conversion=conversion)
    torch.cuda.empty_cache()
    
    
    num_epochs = args.num_epochs
    lr = args.lr
    batch_size = args.batch_size
    device_list = list(map(int, args.devices.split(',')))
    main_device = 'cuda:' + str(device_list[0])
    if args.verbose:
        print(device_list)
    
    for param in model.parameters():
        param.requires_grad = False
    
    if args.lora:
        if args.verbose:
            print('Pre-LoRA:')  
            print(model)
        set_lora(model)
        if args.verbose:
            print('Post-LoRA:')
            print(model)   
     
    lora_params = [param for name, param in model.named_parameters() if 'lora' in name]                   

    adapter = MultiClassifier(768, num_classes=len(conversion))
    
    for param in adapter.parameters():
        param.requires_grad = True    

    parameters = [{"params" : lora_params, "lr": lr/10},
                {"params" : adapter.parameters(), "lr": lr}]
                

    model = AdapterStack(model, 768, adapter).to(main_device)
    optimizer = AdamWScheduleFree(parameters, lr=lr)
    train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    if args.unweighted_loss:
        criterion = torch.nn.CrossEntropyLoss()
    else: 
        weights = torch.ones(len(conversion))
        totals = torch.unique(training_dataset.labels, return_counts=True)[1]
        sum = totals.sum()
        for i in range(len(totals)):
            weights[i] = sum / (len(totals) * totals[i])
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
    model = torch.nn.DataParallel(model, device_ids=device_list)
    trainer = Trainer(model, optimizer, criterion, device_list)
    for i in range(num_epochs):
        trainer.train(train_loader)
        trainer.validate(val_loader)
    data, _, (labels, _) = parse_multiclass_fa('./data/150bp_multiviral_test.fa', class_names=conversion)
    test_dataset = TokenizedDataset(data, labels, tokenizer, conversion=conversion)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    trainer.load_best()
    trainer.test(test_loader)
    
if __name__ == '__main__':
    opt = ArgumentParser()
    opt.add_argument('--batch_size', type=int, default=128)
    opt.add_argument('--lr', type=float, default=1e-3)
    opt.add_argument('--num_epochs', type=int, default=1)
    opt.add_argument('--lora', type=bool, default=True)
    opt.add_argument('--unweighted_loss', action='store_false')
    opt.add_argument('--save_path', type=str, default='./models')
    opt.add_argument('--devices', type=str, default='4,5,6,7')
    opt.add_argument('--verbose', type=bool, default=True)
    args=opt.parse_args()
    main(args)