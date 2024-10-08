import torch
from tqdm import tqdm
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, confusion_matrix
import os
import wandb
class dummy_optimizer():
    def __init__(self):
        pass
    
    def eval(self):
        pass
    
    def train(self):
        pass

class Trainer():
    def __init__(self, model, optimizer, criterion, device_list, args):
        if args.experiment is not None:
            self.save_path = os.path.join(args.save_path, args.experiment)
        else:
            self.save_path = args.save_path
        self.model = model
        self.optimizer = optimizer
        if optimizer is None:
            self.optimizer = dummy_optimizer() # dummy optimizer for inference
        
        self.criterion = criterion
        self.device_list = device_list
        self.conversion = None
        self.epoch = 0
        self.logging_interval = 200
        self.device = 'cuda:' + str(device_list[0])
        self.best_epoch = 0
        self.best_val_loss = 999
        self.epochs_since_best = 0
        self.debug = args.debug
        self.num_classes = args.num_classes
        self.args = args
        if self.num_classes == 1:
            self.train = self.train_binary
            self.validate = self.validate_binary
            self.test = self.test_binary
        else:
            self.train = self.train_fn
            self.validate = self.validate_fn
            self.test = self.test_fn
        

    def train_fn(self, train_loader):
        self.epoch += 1
        if self.conversion is None:
            self.conversion = train_loader.dataset.conversion
        self.model.train()
        self.optimizer.train()
        running_loss = 0.0
        for i, batch in enumerate(tqdm(train_loader)):
            if self.debug:
                if i >= 20:
                    continue
            input  = batch[0]
            labels = batch[1].to(self.device, dtype=torch.int64)
            for key in input.keys():
                input[key] = input[key].to(self.device)
            match self.args.model:
                case 'dnabert-s':
                    outputs = self.model(input_ids=input['input_ids'], 
                                        attention_mask=input['attention_mask'], 
                                        token_type_ids=input['token_type_ids']
                                        )
                case 'nt':
                    outputs = self.model(input_ids=input['input_ids'], 
                                        attention_mask=input['attention_mask']
                                        )
                case 'hyenadna':
                    # print(input['input_ids'].device)
                    # self.model = self.model.to(self.device)
                    outputs = self.model(input_ids=input['input_ids'])
                case _:
                    outputs = self.model(input_ids=input['input_ids'], 
                                        attention_mask=input['attention_mask'], 
                                        token_type_ids=input['token_type_ids']
                                        )
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            running_loss += loss.item()
            if i % self.logging_interval == self.logging_interval - 1:
                wandb.log({'epoch': self.epoch,
                           'batch': i + 1, 
                           'training_loss': running_loss / self.logging_interval})                
                running_loss = 0.0
            
           

    def test_fn(self, test_loader):
        self.model.eval()
        self.optimizer.eval()
        correct = 0
        total = 0
        if self.conversion is None:
            self.conversion = test_loader.dataset.conversion
        if self.conversion is not None:
            per_class = {self.conversion[i] : [0, 0] for i in range(len(self.conversion))} 
        with torch.no_grad():   
            for i, batch in enumerate(test_loader):
                input = batch[0]
                for key in input.keys():
                    input[key] = input[key].to(self.device)
                match self.args.model:
                    case 'dnabert-s':
                        outputs = self.model(input_ids=input['input_ids'], 
                                            attention_mask=input['attention_mask'], 
                                            token_type_ids=input['token_type_ids']
                                            )
                    case 'nt':
                        outputs = self.model(input_ids=input['input_ids'], 
                                            attention_mask=input['attention_mask']
                                            )
                    case 'hyenadna':
                        outputs = self.model(input_ids=input['input_ids'])
                    case _:
                        outputs = self.model(input_ids=input['input_ids'], 
                                            attention_mask=input['attention_mask'], 
                                            token_type_ids=input['token_type_ids']
                                            )
                labels = batch[1].to(self.device, dtype=torch.int64)
                pred_prob = torch.log_softmax(outputs, dim=1)
                predicted = torch.argmax(pred_prob, dim=1)
                batch_y = labels
                if self.conversion is not None:
                    for i in range(len(batch_y)):
                        per_class[self.conversion[batch_y[i]]][1] += 1
                        if predicted[i] == batch_y[i]:
                            per_class[self.conversion[batch_y[i]]][0] += 1
                total += len(input['input_ids'])
                correct += (predicted == batch_y).sum().item()
            
        print(f'Top-1 Accuracy of the network on the test set: {100 * correct / total}%')
        wandb.log({'test_accuracy': 100 * correct / total,
                   'best_epoch': self.best_epoch})
        if self.conversion is not None:
            print(per_class)
            print('Per-class accuracy:')
            for key in per_class:
                print(f'{key}: {100 * per_class[key][0] / per_class[key][1]}')
                wandb.log({f'{key}_accuracy': 100 * per_class[key][0] / per_class[key][1]})
        
            
                
        
    def validate_fn(self, val_loader):
        self.model.eval()
        self.optimizer.eval()
        correct = 0
        total = 0
        running_loss = 0.0
        with torch.no_grad():
            for i, batch in tqdm(enumerate(val_loader)):
                if self.debug:
                    if i >= 20:
                        continue
                input = batch[0]
                for key in input.keys():
                    input[key] = input[key].to(self.device)
                match self.args.model:
                    case 'dnabert-s':
                        outputs = self.model(input_ids=input['input_ids'], 
                                            attention_mask=input['attention_mask'], 
                                            token_type_ids=input['token_type_ids']
                                            )
                    case 'nt':
                        outputs = self.model(input_ids=input['input_ids'], 
                                            attention_mask=input['attention_mask']
                                            )
                    case 'hyenadna':
                        outputs = self.model(input_ids=input['input_ids'])
                    case _:
                        outputs = self.model(input_ids=input['input_ids'], 
                                            attention_mask=input['attention_mask'], 
                                            token_type_ids=input['token_type_ids']
                                            )
                labels = batch[1].to(self.device, dtype=torch.int64)
                
                running_loss += self.criterion(outputs, labels).item()
                pred_prob = torch.log_softmax(outputs, dim=1)
                predicted = torch.argmax(pred_prob, dim=1)
                batch_y = labels
                total += len(input['input_ids'])
                correct += (predicted == batch_y).sum().item()
                
        print(f'Top-1 Accuracy of the network on the validation set: {100 * correct / total}%')
        wandb.log({'validation_accuracy': 100 * correct / total,
                   'validation_loss' : running_loss / total,
                   'epoch': self.epoch})
        if running_loss / total < self.best_val_loss:
            self.best_val_loss = running_loss / total
            self.best_epoch = self.epoch
            self.save_best()
            self.epochs_since_best = 0
        else:
            self.epochs_since_best += 1
        
            
    def train_binary(self, train_loader):
        self.epoch += 1
        if self.conversion is None:
            self.conversion = train_loader.dataset.conversion
        self.model.train()
        self.optimizer.train()
        running_loss = 0.0
        for i, batch in enumerate(tqdm(train_loader)):
            input = batch[0]
            labels = batch[1].to(self.device)
            for key in input.keys():
                input[key] = input[key].to(self.device)
            match self.args.model:
                case 'dnabert-s':
                    outputs = self.model(input_ids=input['input_ids'], 
                                        attention_mask=input['attention_mask'], 
                                        token_type_ids=input['token_type_ids']
                                        )
                case 'nt':
                    outputs = self.model(input_ids=input['input_ids'], 
                                        attention_mask=input['attention_mask']
                                        )
                case 'hyenadna':
                    outputs = self.model(input_ids=input['input_ids'])
                case _:
                    outputs = self.model(input_ids=input['input_ids'], 
                                        attention_mask=input['attention_mask'], 
                                        token_type_ids=input['token_type_ids']
                                        )
            outputs = outputs.squeeze()
            if outputs.size() != labels.size():
                print('squeezed labels')
                labels = labels.squeeze()
                if outputs.size() != labels.size():
                    print('failed')
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            running_loss += loss.item()
            if i % self.logging_interval == self.logging_interval - 1:
                wandb.log({'epoch': self.epoch,
                           'batch': i + 1, 
                           'training_loss': (running_loss / self.logging_interval)})
                # print(f'[{self.epoch}, {i + 1}] loss: {running_loss / self.logging_interval}')
                running_loss = 0.0
           

    def test_binary(self, test_loader):
        self.model.eval()
        self.optimizer.eval()
        correct = 0
        total = 0
        pred_probs = []
        y_true = []
        preds = []
        if self.conversion is not None:
            per_class = {self.conversion[i] : [0, 0] for i in range(len(self.conversion))}    
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if self.debug:
                    if i >= 20:
                        continue
                input = batch[0]
                for key in input:
                    input[key] = input[key].to(self.device)
                labels = batch[1].to(self.device)
                match self.args.model:
                    case 'dnabert-s':
                        outputs = self.model(input_ids=input['input_ids'], 
                                            attention_mask=input['attention_mask'], 
                                            token_type_ids=input['token_type_ids']
                                            )
                    case 'nt':
                        outputs = self.model(input_ids=input['input_ids'], 
                                            attention_mask=input['attention_mask']
                                            )
                    case 'hyenadna':
                        outputs = self.model(input_ids=input['input_ids'])
                    case _:
                        outputs = self.model(input_ids=input['input_ids'], 
                                            attention_mask=input['attention_mask'], 
                                            token_type_ids=input['token_type_ids']
                                            )
                outputs = outputs.squeeze()
                pred_prob = F.sigmoid(outputs)
                predicted = torch.tensor([1 if pred_prob[i] > 0.5 else 0 for i in range(len(pred_prob))]).to(self.device)
                batch_y = labels
                y_true.extend(batch_y.tolist())
                pred_probs.extend(pred_prob.tolist())
                preds.extend(predicted.tolist())
                total += len(labels)
                correct += (predicted == batch_y).sum().item()
        print(f'Binary Accuracy of the network on the test set: {100 * correct / total}%')
        print(f'ROC AUC Score: {roc_auc_score(y_true, pred_probs)}')
        cm = confusion_matrix(y_true, preds)
        wandb.log({'test_accuracy': 100 * correct / total,
                     'roc_auc_score': roc_auc_score(y_true, pred_probs),
                     'TPR': cm[1][1] / (cm[1][1] + cm[1][0]),
                     'FPR': cm[0][1] / (cm[0][1] + cm[0][0]),
                     'best_epoch': self.best_epoch})
        return (y_true, pred_probs)
                
        
    def validate_binary(self, val_loader):
        self.model.eval()
        self.optimizer.eval()
        correct = 0
        total = 0
        running_loss = 0.0
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if self.debug:
                    if i >= 20:
                        continue
                input = batch[0]
                for key in input:
                    input[key] = input[key].to(self.device)
                labels = batch[1].to(self.device)
                
                match self.args.model:
                    case 'dnabert-s':
                        outputs = self.model(input_ids=input['input_ids'], 
                                            attention_mask=input['attention_mask'], 
                                            token_type_ids=input['token_type_ids']
                                            )
                    case 'nt':
                        outputs = self.model(input_ids=input['input_ids'], 
                                            attention_mask=input['attention_mask']
                                            )
                    case 'hyenadna':
                        outputs = self.model(input_ids=input['input_ids'])
                    case _:
                        outputs = self.model(input_ids=input['input_ids'], 
                                            attention_mask=input['attention_mask'], 
                                            token_type_ids=input['token_type_ids']
                                            )
                outputs = outputs.squeeze()
                running_loss += self.criterion(outputs, labels).item()
                pred_prob = F.sigmoid(outputs)
                predicted = torch.tensor([1 if pred_prob[i] > 0.5 else 0 for i in range(len(pred_prob))]).to(self.device)
                batch_y = labels
                total += len(labels)
                correct += (predicted == batch_y).sum().item()
                
        print(f'Binary Accuracy of the network on the validation set: {100 * correct / total}%')
        wandb.log({'validation_accuracy': 100 * correct / total,
                   'validation_loss' : running_loss / total,
                   'epoch': self.epoch})
        if running_loss / total < self.best_val_loss:
            self.best_val_loss = running_loss / total
            self.best_epoch = self.epoch
            self.save_best()
            self.epochs_since_best = 0
        else:
            self.epochs_since_best += 1
                    
    def save(self):
        if isinstance(len(self.device_list) > 1):
            print("SAVING FROM DP")
            torch.save(self.model.module.state_dict(), os.path.join(self.save_path, f"model_{self.epoch}.pth"))
        else:
           torch.save(self.model.state_dict(), os.path.join(self.save_path, f"model_{self.epoch}.pth"))
        
    def save_best(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        # self.save()
        if len(self.device_list) > 1:
            print("SAVING FROM DP")
            torch.save(self.model.module.state_dict(), os.path.join(self.save_path, f"model_{self.epoch}.pth"))
        else:
           torch.save(self.model.state_dict(), os.path.join(self.save_path, f"model_{self.epoch}.pth"))
        
                
    def load(self, epoch):
        if len(self.device_list) > 1:
            self.model.module.load_state_dict(torch.load(os.path.join(self.save_path, 'model_' + str(epoch) + '.pth')))
        else:
            self.model.load_state_dict(torch.load(os.path.join(self.save_path, 'model_' + str(epoch) + '.pth')))
        
    def load_best(self):
        print(f'loading best model at epoch {self.best_epoch}')
        
        if len(self.device_list) > 1:
            self.model.module.load_state_dict(torch.load(os.path.join(self.save_path, 'best_model.pth')))
        else:
            self.model.load_state_dict(torch.load(os.path.join(self.save_path, 'best_model.pth')))
        