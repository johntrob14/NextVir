import torch
from schedulefree import AdamWScheduleFree
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModel
from utils import Trainer, parse_multiclass_fa, parse_HPV_fa, TokenizedDataset, get_stack, get_criterion
import wandb
from torch.nn import functional as F

def main(args):
    torch.manual_seed(args.seed)
    
    # Initialize Tokenizer and Base Model
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-S", trust_remote_code=True)
    model = AutoModel.from_pretrained("zhihan1996/DNABERT-S", trust_remote_code=True)        
    
    # Parse fasta datasets
    data, bin_labels, (labels, conversion) = parse_multiclass_fa('./data/150bp_multiviral_train.fa')
    # conversion = ['HUM', 'HPV']
    # data, bin_labels = parse_HPV_fa('./data/HPV/reads_150_train.fa')
    
    torch.cuda.empty_cache()
    
    
    num_epochs = args.num_epochs
    lr = args.lr
    batch_size = args.batch_size
    device_list = list(map(int, args.device.split(',')))
    
    args.main_device = 'cuda:' + str(device_list[0]) # add to args for ease of use
    if args.verbose:
        print(device_list)
    
    model, parameters = get_stack(model, args)
    
    print('model_got')
    # Loaders and criterion
    
    if len(device_list) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_list)
    model.load_state_dict(torch.load('./models/HHV-8_vs_all_bin/best_model.pth'))
    print('model_loaded')
    # Train and Validate
    # trainer = Trainer(model, optimizer, criterion, device_list, args)
    # for i in range(num_epochs):
    #     trainer.train(train_loader)
    #     trainer.validate(val_loader)
    
    # Test
    data, bin_labels, (labels, _) = parse_multiclass_fa('./data/150bp_multiviral_test.fa', class_names=conversion)
    test_dataset = TokenizedDataset(data, labels, tokenizer, conversion=conversion)
    test_dataset.subsample_one_vs_all = one_class_subsample_one_vs_all
    test_dataset.subsample_one_vs_all(test_dataset, args.single_label)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    acc_test_binary(model, args.main_device, test_loader)
    
def one_class_subsample_one_vs_all(self, label):
    new_reads = []
    for i in range(len(self.labels)):
        if self.labels[i] == self.conversion.index(label):
            new_reads.append(self.reads[i])
    self.reads = torch.stack(new_reads)
    self.labels = torch.ones(len(new_reads))

    
def acc_test_binary(model, main_device, test_loader):
    model.eval()
    correct = 0
    total = 0
    pred_probs = []
    y_true = [] 
    for batch in test_loader:
        input = batch[0]
        for key in input:
            input[key] = input[key].to(main_device)
        labels = batch[1].to(main_device)
        outputs = model(input_ids=input['input_ids'], attention_mask=input['attention_mask'], token_type_ids=input['token_type_ids']).squeeze()
        pred_prob = F.sigmoid(outputs)
        predicted = torch.tensor([1 if pred_prob[i] > 0.5 else 0 for i in range(len(pred_prob))]).to(main_device)
        batch_y = labels
        y_true.extend(batch_y.tolist())
        pred_probs.extend(pred_prob.tolist())
        total += len(labels)
        correct += (predicted == batch_y).sum().item()
    print(f'Binary Accuracy of the network on the test set: {100 * correct / total}%')
    wandb.log({'test_accuracy': 100 * correct / total})
    # TODO: Implement per-class accuracy on the binary set (not sur if this is actually needed)

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