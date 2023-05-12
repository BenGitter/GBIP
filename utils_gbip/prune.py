import torch
from torch.nn import Upsample
from models.common import Conv, Concat, MP, SP
from models.yolo import Detect

def index_remove(tensor, dim, index, removed=False):
    if tensor.is_cuda:
        tensor = tensor.cpu()
    size_ = list(tensor.size())
    new_size = tensor.size(dim) - len(index)
    size_[dim] = new_size
    new_size = size_

    select_index = list(set(range(tensor.size(dim))) - set(index))
    new_tensor = torch.index_select(tensor, dim, torch.tensor(select_index))
    
    if removed:
        return new_tensor, torch.index_select(tensor, dim, torch.tensor(index))

    return new_tensor

def get_new_conv(conv, dim, channel_index, independent_prune_flag=False):
    has_bias = conv.bias is not None
    if dim == 0:
        new_conv = torch.nn.Conv2d(in_channels=conv.in_channels,
                                   out_channels=int(conv.out_channels - len(channel_index)),
                                   kernel_size=conv.kernel_size, bias=has_bias,
                                   stride=conv.stride, padding=conv.padding, dilation=conv.dilation)
        
        new_conv.weight.data = index_remove(conv.weight.data, dim, channel_index)
        if conv.bias is not None:
            new_conv.bias.data = index_remove(conv.bias.data, dim, channel_index)
    elif dim == 1:
        new_conv = torch.nn.Conv2d(in_channels=int(conv.in_channels - len(channel_index)),
                                   out_channels=conv.out_channels,
                                   kernel_size=conv.kernel_size, bias=has_bias,
                                   stride=conv.stride, padding=conv.padding, dilation=conv.dilation)
        
        new_weight = index_remove(conv.weight.data, dim, channel_index, independent_prune_flag)
        residue = None
        if independent_prune_flag:
            new_weight, residue = new_weight
        new_conv.weight.data = new_weight
        if conv.bias is not None:
            new_conv.bias.data = conv.bias.data

    return new_conv
    
def get_new_norm(norm, channel_index):
    new_norm = torch.nn.BatchNorm2d(num_features=int(norm.num_features - len(channel_index)),
                                    eps=norm.eps,
                                    momentum=norm.momentum,
                                    affine=norm.affine,
                                    track_running_stats=norm.track_running_stats)

    new_norm.weight.data = index_remove(norm.weight.data, 0, channel_index)
    new_norm.bias.data = index_remove(norm.bias.data, 0, channel_index)

    if norm.track_running_stats:
        new_norm.running_mean.data = index_remove(norm.running_mean.data, 0, channel_index)
        new_norm.running_var.data = index_remove(norm.running_var.data, 0, channel_index)
        
    return new_norm
    
def replace_nonconv(model, idx):
    # print('[replace_concat]', idx)
    new_idx = []
    for i in idx:
        if type(model.model[i]) in [Concat, MP, SP, Upsample]:
        # if isinstance(model.model[i], Concat) or isinstance(model.model[i], MP) or isinstance(model.model[i], SP):
            prev = model.model[i].f
            if isinstance(prev, int):
                prev = [prev]
            for f in prev:
                if f < 0:
                    f += model.model[i].i
                new_idx.append(f)
        else:
            new_idx.append(i)
    if idx == new_idx:
        return new_idx
    else:
        return replace_nonconv(model, new_idx)

def find_removed_in_channels(model, l):
    cin_remove = []
    if l.i > 0:
        l_f = l.f
        l_prev = []
        if isinstance(l_f, int):
            l_f = [l_f]  
        for f in l_f:
            if f < 0:
                l_prev.append(l.i + f)
            else:
                l_prev.append(f)
        l_prev = replace_nonconv(model, l_prev)
        
        if len(l_prev) > 1:
            sum_out = 0
            for f in l_prev:
                l_f = model.model[f]
                cin_remove += [x + sum_out for x in l_f.cout_removed]
                sum_out += l_f.conv.out_channels + len(l_f.cout_removed)
        else:
            cin_remove = model.model[l_prev[0]].cout_removed
    return cin_remove

def fix_input_detect(model, idx):
    l = model.model[idx]
    for i in range(l.nl):
        cin_remove = model.model[l.f[i]].cout_removed
        l.m[i] = get_new_conv(l.m[i], 1, cin_remove)

@torch.no_grad()
def prune_step(model, img_batch, k, device):
    model.eval()
    model.to(device)
    x = img_batch.to(device).float() / 255.0

    # calculate which channels to remove from each layer
    for l in model.model:
        # ignore anything but Conv layers
        if not isinstance(l, Conv):
            continue
        
        y = model.forward_till_layer(x, l)

        # calculate importance score and threshold
        m_l = torch.norm(y.detach(), 1, (0,2,3))
        m_l = m_l / torch.max(m_l)
        m_l_p = k * torch.sum(m_l) / l.conv.out_channels
        cout_remove = (m_l < m_l_p).nonzero().squeeze(1).tolist()
        print(cout_remove)

        # set indices of channels to be removed
        l.cout_remove = cout_remove

    # actually remove channels and make sure input channels still match
    l_detect = 0
    for l in model.model:
        # ignore anything but Conv layers and save Detect() layer number
        if not isinstance(l, Conv):
            if isinstance(l, Detect):
                l_detect = l.i
            continue

        # first check if previous layers changed output and make sure input matches this change
        cin_remove = find_removed_in_channels(model, l)
        l.conv = get_new_conv(l.conv, 1, cin_remove)
        
        # remove output channels; calculated above
        l.conv = get_new_conv(l.conv, 0, l.cout_remove)
        l.bn = get_new_norm(l.bn, l.cout_remove)
        l.cout_removed = l.cout_remove
        del l.cout_remove
    
    # make sure input channels of Detect() still match
    fix_input_detect(model, l_detect)

    # test full model
    model.to(device)
    y = model(x)