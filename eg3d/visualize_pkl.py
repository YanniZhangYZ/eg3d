import pickle
import torch
import io
import legacy
import dnnlib
import json

def named_params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.named_parameters()) + list(module.named_buffers())

def copy_params_and_buffers(src_module, dst_module, require_all=False):
    assert isinstance(src_module, torch.nn.Module)
    assert isinstance(dst_module, torch.nn.Module)
    src_tensors = dict(named_params_and_buffers(src_module))
    for name, tensor in named_params_and_buffers(dst_module):
        assert (name in src_tensors) or (not require_all)
        if name in src_tensors:
            tensor.copy_(src_tensors[name].detach()).requires_grad_(tensor.requires_grad)


device = torch.device('cpu')
with dnnlib.util.open_url("ffhqrebalanced512-128.pkl") as f:
   
    data = legacy.load_network_pkl(f)
    G = data['G'].to(device)# type: ignore
    
    G_em = data['G_ema'].to(device) # type: ignore
    D = data['D'].to(device)
    # train_set = data['training_set_kwargs'].to(device) # type: ignore
    # augment_pipe = data['augment_pipe'].to(device)
    
    print(G)
    print("--------------------------------------------")
    print(G_em)
    print("--------------------------------------------")
    # print(D)
