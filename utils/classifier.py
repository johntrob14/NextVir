import torch
from torch import nn
from sklearn.metrics import roc_auc_score

class AdapterStack(nn.Module):
    def __init__(self, base_model, adapter, args):
        super(AdapterStack, self).__init__()
        self.base_model = base_model
        self.adapter = adapter
        self.model_type = args.model
        # match args.model:
        
        
    def forward_ds(self, input_ids, attention_mask=None, token_type_ids=None):
        x = self.base_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0] # D-S
        # x = self.base_model(input_ids=input_ids, attention_mask=attention_mask, encoder_attention_mask=attention_mask, output_hidden_states=True)['hidden_states'][-1] NT
        attention_mask = torch.unsqueeze(attention_mask, dim=-1)

        # Compute mean embeddings per sequence
        x = torch.sum(attention_mask*x, axis=-2)/torch.sum(attention_mask, axis=1)
        x = self.adapter(x)
        return x     
    
    def forward_nt(self, input_ids, attention_mask=None):
        x = self.base_model(input_ids=input_ids, attention_mask=attention_mask, encoder_attention_mask=attention_mask, output_hidden_states=True)['hidden_states'][-1] # NT
        x = x.mean(dim=1)
        x = self.adapter(x)
        return x
    
    def forward_hdna(self, input_ids):
        for name, p in self.base_model.named_parameters():
            if p.device != input_ids.device:
                print(name, p.device, input_ids.device)
                # p = p.to(input_ids.device)
        x = self.base_model(input_ids=input_ids, output_hidden_states=True)['hidden_states'][-1] # H-DNA
        x = x.mean(dim=1)
        x = self.adapter(x)
        return x
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        match self.model_type:
            case 'dnabert-s':
                x = self.forward_ds(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            case 'nt':
                x = self.forward_nt(input_ids, attention_mask=attention_mask)
            case 'hyenadna':
                x = self.forward_hdna(input_ids)
            case _:
                x = self.forward_ds(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
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