import torch
from transformers import AutoTokenizer, AutoModel
from .lora import LoRaModule
from torch.nn.utils.parametrize import register_parametrization

# recursively search and set parameterization for all Wqkv matrices to LoraMerged layers
def set_lora(model):
    for name, module in model.named_children():
        if 'Wqkv' in name:
            register_parametrization(module, 'weight', LoRaModule(*module.weight.shape, split_dimension=3, rank=4, device=module.weight.device))
        elif isinstance(module, torch.nn.Dropout):
            module.p = 0.0
        elif isinstance(module, torch.nn.Linear):
            register_parametrization(module, 'weight', LoRaModule(*module.weight.shape, rank=4, device=module.weight.device))
        else:
            set_lora(module)

if __name__ == '__main__':
    
    model = AutoModel.from_pretrained("zhihan1996/DNABERT-S", trust_remote_code=True)
    print(model)
    set_lora(model)
    print(model)
    # print(model)
    # print(model.config)
    # print(model.state_dict().keys())
    # for name, module in model.named_children():
    #     print(name)
    #     print(module)