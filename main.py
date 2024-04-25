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
    
    
    num_epochs = 1
    batch_size = 128
    lr = 1e-3
    for param in model.parameters():
        param.requires_grad = False
        
    print(model)
    set_lora(model)
    print(model)   
     
    lora_params = [param for name, param in model.named_parameters() if 'lora' in name]                   

    adapter = MultiClassifier(768, num_classes=len(conversion))
    
    for param in adapter.parameters():
        param.requires_grad = True    

    parameters = [{"params" : lora_params, "lr": lr/10},
                {"params" : adapter.parameters(), "lr": lr}]
                

    model = AdapterStack(model, 768, adapter).to('cuda:4')
    optimizer = AdamWScheduleFree(parameters, lr=lr)
    train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()
    device_list = [4,5,6,7]
    model = torch.nn.DataParallel(model, device_ids=device_list)
    trainer = Trainer(model, optimizer, criterion, device_list)
    for i in range(num_epochs):
        trainer.train(train_loader)
        trainer.validate(val_loader)
        trainer.save()
    data, _, (labels, _) = parse_multiclass_fa('./data/150bp_multiviral_test.fa', class_names=conversion)
    test_dataset = TokenizedDataset(data, labels, tokenizer, conversion=conversion)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    trainer.load_best()
    trainer.test(test_loader)
    
if __name__ == '__main__':
    opt = ArgumentParser()
    
    
    
    
    args=opt.parse_args()
    main(args)