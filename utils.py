import torch
import os
import torch.nn as nn

def restore_checkpoint(dir_ckpt):
    state = torch.load(dir_ckpt)
    return state

def save_checkpoint(dir, state):
    ckpt = {
            'net_model': state['net_model'],
            'sched': state['sched'],
            'optim': state['optim'],
            'step': state['step'],
            'ema_model': state['ema_model'],
            'cemblayer': state['cemblayer']
            }
    torch.save(ckpt, os.path.join(dir, 'ckpt_{}.pt'.format(state['step'])))
    torch.save(ckpt, os.path.join(dir, 'ckpt.pt'))



# def ema(source, target, decay):
#     source_dict = source.state_dict()
#     target_dict = target.state_dict()
#     for key in source_dict.keys():
#         target_dict[key].data.copy_(
#             target_dict[key].data * decay +
#             source_dict[key].data * (1 - decay))

# from https://github.com/ermongroup/ddim/blob/main/models/ema.py
class EMAHelper(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (
                    1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(
                inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        # module_copy = copy.deepcopy(module)
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict

class ConditionalEmbedding(nn.Module):
    def __init__(self, num_labels:int, d_model:int, dim:int):
        assert d_model % 2 == 0
        super().__init__()
        self.condEmbedding = nn.Sequential(
            nn.Embedding(num_embeddings=num_labels + 1, embedding_dim=d_model, padding_idx=0),
            nn.Linear(d_model, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t:torch.Tensor) -> torch.Tensor:
        emb = self.condEmbedding(t)
        return emb

