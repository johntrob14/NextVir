import torch

from schedulefree import AdamWScheduleFree
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModel
from utils import set_lora, BinaryTrainer, parse_multiclass_fa, TokenizedDataset, BinClassifier, AdapterStack

def main(args):
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-S", trust_remote_code=True)
    model = AutoModel.from_pretrained("zhihan1996/DNABERT-S", trust_remote_code=True)
    data, labels, (_, conversion) = parse_multiclass_fa('./data/150bp_multiviral_train.fa')
    training_dataset = TokenizedDataset(data, labels, tokenizer, conversion=conversion)
    data, labels, (_, _) = parse_multiclass_fa('./data/150bp_multiviral_val.fa', class_names=conversion)
    val_dataset = TokenizedDataset(data, labels, tokenizer, conversion=conversion)
    torch.cuda.empty_cache()
    
    
    num_epochs = args.num_epochs
    lr = args.lr
    batch_size = args.batch_size
    device_list = list(map(int, args.device.split(',')))
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
        if args.verbose:
            print('LoRA Parameters:')
            print(lora_params)
        
    
    adapter = BinClassifier(768)
    
    for param in adapter.parameters():
        param.requires_grad = True    

    if not args.embedding_only:
        parameters = [{"params" : lora_params, "lr": lr/10, "warmup_steps": 500},
                    {"params" : adapter.parameters(), "lr": lr}]
    else:
        parameters = [{"params" : adapter.parameters(), "lr": lr}]
                

    model = AdapterStack(model, 768, adapter).to(main_device)
    optimizer = AdamWScheduleFree(parameters, lr=lr, weight_decay=0.004)
    train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    if len(device_list) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_list)
    trainer = BinaryTrainer(model, optimizer, criterion, device_list)
    for i in range(num_epochs):
        trainer.train(train_loader)
        trainer.validate(val_loader)
        
    data, labels, (_, _) = parse_multiclass_fa('./data/150bp_multiviral_test.fa', class_names=conversion)
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
    opt.add_argument('--unweighted_loss', action='store_false')
    opt.add_argument('--save_path', type=str, default='./models')
    opt.add_argument('--device', type=str, default='4,5,6,7')
    opt.add_argument('--verbose', type=bool, default=True)
    args=opt.parse_args()
    print(args.lora)
    main(args)
    
    #TODO: merge this with the other main.py