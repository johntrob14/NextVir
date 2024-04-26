import torch
from torch import nn
from sklearn.metrics import roc_auc_score

class AdapterStack(nn.Module):
    def __init__(self, base_model, input_size, adapter):
        super(AdapterStack, self).__init__()
        self.base_model = base_model
        self.adapter = adapter
        
    def forward(self, **kwargs):
        x = self.base_model(**kwargs)[0]
        x = x.mean(dim=1)
        x = self.adapter(x)
        return x     
        
class BinClassifier(nn.Module):
    def __init__(self, input_size):
        super(BinClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
class MultiClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MultiClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
def train(train_loader, epoch_num, model, mode='binary', verbose=False, device='cuda:3'):
    model.to(device)
    model.train()
    if mode == 'binary':
        criterion = nn.BCELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for i, data in enumerate(train_loader):
        inputs, labels = data['embedding'].to(device), data['labels'].to(device)
        if mode == 'binary':
            labels_binary = torch.ones(len(labels))
            labels_binary[labels[:, 0] == 1] = 0
            labels = labels_binary.to(device)
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if verbose and i % 1000 == 0:
            print(f'Epoch {epoch_num}, Loss: {loss.item()}')
    print('Finished Training')
    
def validate(val_loader, model, mode='binary', conversion=None, device='cuda:3'):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    preds = []
    y = []
    if conversion is not None:
        per_class = {conversion[i] : [0, 0] for i in range(len(conversion))}
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data['embedding'].to(device), data['labels'].to(device)
            
            if mode == 'binary':
                outputs = model(inputs).squeeze()
                preds.extend(outputs.tolist())
                batch_y = [0 if labels[i][0] == 1 else 1 for i in range(len(labels))]
                y.extend(batch_y)
                predicted = (outputs > 0.5)
                if conversion is not None:
                    for i in range(len(labels)):
                        correct_index = torch.argmax(labels[i])
                        per_class[conversion[correct_index]][1] += 1
                        if predicted[i] == batch_y[i]:
                            per_class[conversion[correct_index]][0] += 1
                total += len(batch_y)
                for i in range(len(batch_y)):
                    if predicted[i] == batch_y[i]:
                        correct += 1
            else:
                outputs = model(inputs)
                pred_prob = torch.log_softmax(outputs, dim=1)
                predicted = torch.argmax(pred_prob, dim=1)
                batch_y = labels
                if conversion is not None:
                    for i in range(len(batch_y)):
                        per_class[conversion[int(batch_y[i])]][1] += 1
                        if predicted[i] == batch_y[i]:
                            per_class[conversion[int(batch_y[i])]][0] += 1
                total += len(inputs)
                correct += (predicted == batch_y).sum().item()
                
    
    if mode == 'binary':
        print(f'Accuracy of the network on the validation set: {100 * correct / total}%')
        print(f'AUC of the network on the validation set: {roc_auc_score(y, preds)}')
    else:
        
        print(f'Top-1 Accuracy of the network on the validation set: {100 * correct / total}%')
    if conversion is not None:
        print(per_class)
        print('Per-class accuracy:')
        for key in per_class:
            print(f'{key}: {100 * per_class[key][0] / per_class[key][1]}')
            
def train_weighted(train_loader, epoch_num, model, mode='binary', verbose=False, device='cuda:3'):
    model.to(device)
    model.train()
    if mode == 'binary':
        criterion = nn.BCELoss()
    else:
        weights = torch.ones(len(train_loader.dataset.conversion))
        classes, totals = torch.unique(train_loader.dataset.labels, return_counts=True)
        sum = totals.sum()
        for i in range(len(totals)):
            weights[int(classes[i])] = sum / (len(totals) * totals)[i]
        criterion = torch.nn.CrossEntropyLoss(weight=weights.to('cuda:3'))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for i, data in enumerate(train_loader):
        inputs, labels = data['embedding'].to(device), data['labels'].to(device, dtype=torch.int64)
        if mode == 'binary':
            labels_binary = torch.ones(len(labels))
            labels_binary[labels[:, 0] == 1] = 0
            labels = labels_binary.to(device)
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if verbose and i % 1000 == 0:
            print(f'Epoch {epoch_num}, Loss: {loss.item()}')
    print('Finished Training')