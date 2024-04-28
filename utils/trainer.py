import torch
from tqdm import tqdm
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
import os

class Trainer():
    def __init__(self, model, optimizer, criterion, device_list, args):
        self.save_path = os.path.join(args.save_path, args.experiment)
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device_list = device_list
        self.conversion = None
        self.epoch = 0
        self.logging_interval = 2000
        self.device = 'cuda:' + str(device_list[0])
        self.best_epoch = 0
        self.best_val_loss = 999
        self.debug = args.debug
        

    def train(self, train_loader):
        self.epoch += 1
        if self.conversion is None:
            self.conversion = train_loader.dataset.conversion
        self.model.train()
        self.optimizer.train()
        running_loss = 0.0
        for i, batch in enumerate(tqdm(train_loader)):
            input = batch[0]
            labels = batch[1].to(self.device, dtype=torch.int64)
            for key in input.keys():
                input[key] = input[key].to(self.device)
            outputs = self.model(input_ids=input['input_ids'], attention_mask=input['attention_mask'], token_type_ids=input['token_type_ids'])
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            running_loss += loss.item()
            if i % self.logging_interval == self.logging_interval - 1:
                print(f'[{self.epoch}, {i + 1}] loss: {running_loss / self.logging_interval}')
                running_loss = 0.0
            if self.debug:
                if i == 200:
                    break
           

    def test(self, test_loader):
        self.model.eval()
        self.optimizer.eval()
        correct = 0
        total = 0
        if self.conversion is not None:
            per_class = {self.conversion[i] : [0, 0] for i in range(len(self.conversion))}    
        for batch in test_loader:
            input = batch[0]
            for key in input.keys():
                input[key] = input[key].to(self.device)
            outputs = self.model(input_ids=input['input_ids'], attention_mask=input['attention_mask'], token_type_ids=input['token_type_ids'])
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
        if self.conversion is not None:
            print(per_class)
            print('Per-class accuracy:')
            for key in per_class:
                print(f'{key}: {100 * per_class[key][0] / per_class[key][1]}')
                
        
    def validate(self, val_loader):
        self.model.eval()
        self.optimizer.eval()
        correct = 0
        total = 0
        running_loss = 0.0
        for batch in val_loader:
            input = batch[0]
            for key in input.keys():
                input[key] = input[key].to(self.device)
            outputs = self.model(input_ids=input['input_ids'], attention_mask=input['attention_mask'], token_type_ids=input['token_type_ids'])
            labels = batch[1].to(self.device, dtype=torch.int64)
            
            running_loss += self.criterion(outputs, labels).item()
            pred_prob = torch.log_softmax(outputs, dim=1)
            predicted = torch.argmax(pred_prob, dim=1)
            batch_y = labels
            total += len(input['input_ids'])
            correct += (predicted == batch_y).sum().item()
            
        print(f'Top-1 Accuracy of the network on the validation set: {100 * correct / total}%')
        if running_loss / total < self.best_val_loss:
            self.best_val_loss = running_loss / total
            self.best_epoch = self.epoch
            self.save_best()
                    
    def save(self):
        torch.save(self.model.state_dict(), os.path.join(self.save_path, f"model_{self.epoch}.pth"))
        
    def save_best(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        torch.save(self.model.state_dict(), os.path.join(self.save_path, 'best_model.pth'))
        
    def load(self, epoch):
        self.model.load_state_dict(torch.load(os.path.join(self.save_path, 'model_' + str(epoch) + '.pth')))
        
    def load_best(self):
        print(f'loading best model at epoch {self.best_epoch}')
        self.model.load_state_dict(torch.load(os.path.join(self.save_path,'best_model.pth')))
        
class BinaryTrainer():
    #TODO: maybe? combine this with the multiclass trainer
    def __init__(self, model, optimizer, criterion, device_list):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device_list = device_list
        self.conversion = None
        self.epoch = 0
        self.logging_interval = 2000
        self.device = 'cuda:' + str(device_list[0])
        self.best_epoch = 0
        self.best_val_loss = 999
        

    def train(self, train_loader):
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
            outputs = self.model(input_ids=input['input_ids'], attention_mask=input['attention_mask'], token_type_ids=input['token_type_ids']).squeeze()
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            running_loss += loss.item()
            if i % self.logging_interval == self.logging_interval - 1:
                print(f'[{self.epoch}, {i + 1}] loss: {running_loss / self.logging_interval}')
                running_loss = 0.0
           

    def test(self, test_loader):
        self.model.eval()
        self.optimizer.eval()
        correct = 0
        total = 0
        pred_probs = []
        y_true = []
        if self.conversion is not None:
            per_class = {self.conversion[i] : [0, 0] for i in range(len(self.conversion))}    
        for batch in test_loader:
            input = batch[0]
            for key in input:
                input[key] = input[key].to(self.device)
            labels = batch[1].to(self.device)
            outputs = self.model(input_ids=input['input_ids'], attention_mask=input['attention_mask'], token_type_ids=input['token_type_ids']).squeeze()
            pred_prob = F.sigmoid(outputs)
            predicted = torch.tensor([1 if pred_prob[i] > 0.5 else 0 for i in range(len(pred_prob))]).to(self.device)
            batch_y = labels
            y_true.extend(batch_y.tolist())
            pred_probs.extend(pred_prob.tolist())
            total += len(labels)
            correct += (predicted == batch_y).sum().item()
        print(f'Binary Accuracy of the network on the test set: {100 * correct / total}%')
        print(f'ROC AUC Score: {roc_auc_score(y_true, pred_probs)}')
        # TODO: Implement per-class accuracy on the binary set (not sur if this is actually needed)
                
        
    def validate(self, val_loader):
        self.model.eval()
        self.optimizer.eval()
        correct = 0
        total = 0
        running_loss = 0.0
        for batch in val_loader:
            input = batch[0]
            for key in input:
                input[key] = input[key].to(self.device)
            labels = batch[1].to(self.device)
            
            outputs = self.model(input_ids=input['input_ids'], attention_mask=input['attention_mask'], token_type_ids=input['token_type_ids']).squeeze()
            running_loss += self.criterion(outputs, labels).item()
            pred_prob = F.sigmoid(outputs)
            predicted = torch.tensor([1 if pred_prob[i] > 0.5 else 0 for i in range(len(pred_prob))]).to(self.device)
            batch_y = labels
            total += len(labels)
            correct += (predicted == batch_y).sum().item()
            
        print(f'Binary Accuracy of the network on the validation set: {100 * correct / total}%')
        if running_loss / total < self.best_val_loss:
            self.best_val_loss = running_loss / total
            self.best_epoch = self.epoch
            self.save_best()
                    
    def save(self):
        torch.save(self.model.state_dict(), f"./models/model_{self.epoch}.pth")
        
    def save_best(self):
        torch.save(self.model.state_dict(), f"./models/bin/best_adapter.pth")
        
    def load(self, epoch):
        self.model.load_state_dict(torch.load('./models/model_' + str(epoch) + '.pth'))
        
    def load_best(self):
        print(f'loading best model at epoch {self.best_epoch}')
        self.model.load_state_dict(torch.load('./models/bin/best_adapter.pth'))
    