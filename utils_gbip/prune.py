import torch
from torch.nn import Upsample
from models.common import Conv, Concat, MP, SP, SPPCSPC, RepConv, ImplicitA
from models.yolo import IDetect

from tqdm import tqdm

# Based on: https://github.com/tyui592/Pruning_filters_for_efficient_convnets/blob/master/prune.py

def get_removed_channels(model, layers):
    c = []
    for i in layers:
        c.append(model.model[i].remove_hist)
    return c

def index_remove(tensor, dim, index, removed=False):
    # if tensor.is_cuda:
    #     tensor = tensor.cpu()
    size_ = list(tensor.size())
    new_size = tensor.size(dim) - len(index)
    size_[dim] = new_size
    new_size = size_

    select_index = torch.tensor(list(set(range(tensor.size(dim))) - set(index))).cuda()
    new_tensor = torch.index_select(tensor, dim, select_index)
    
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

def get_new_ia(ia, channel_index):
    new_ia = ImplicitA(int(ia.channel - len(channel_index)), ia.mean, ia.std)
    new_ia.implicit.data = index_remove(ia.implicit.data, 1, channel_index)

    return new_ia


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
                if isinstance(l_f, SPPCSPC): # FIX
                    sum_out += l_f.cv7.conv.out_channels
                else:
                    sum_out += l_f.conv.out_channels + len(l_f.cout_removed)
        else:
            cin_remove = model.model[l_prev[0]].cout_removed
    return cin_remove

def fix_input_idetect(model, idx):
    l = model.model[idx]
    for i in range(l.nl):
        cin_remove = model.model[l.f[i]].cout_removed
        l.m[i] = get_new_conv(l.m[i], 1, cin_remove)
        l.ia[i] = get_new_ia(l.ia[i], cin_remove)

def prune_repconv(model, l):
    # first check if previous layers changed output and make sure input matches this change
    cin_remove = find_removed_in_channels(model, l)
    l.rbr_dense[0] = get_new_conv(l.rbr_dense[0], 1, cin_remove)
    l.rbr_1x1[0] = get_new_conv(l.rbr_1x1[0], 1, cin_remove)

    # remove output channels; calculated above
    l.rbr_dense[0] = get_new_conv(l.rbr_dense[0], 0, l.cout_remove)
    l.rbr_dense[1] = get_new_norm(l.rbr_dense[1], l.cout_remove)
    l.rbr_1x1[0] = get_new_conv(l.rbr_1x1[0], 0, l.cout_remove)
    l.rbr_1x1[1] = get_new_norm(l.rbr_1x1[1], l.cout_remove)
    l.cout_removed = l.cout_remove
    del l.cout_remove

    if not hasattr(l, 'remove_hist'):
        l.remove_hist = []
        l.remove_hist.append(l.cout_removed)

    # update yaml
    l.out_channels = l.rbr_dense[0].out_channels
    model.yaml['head'][l.i-len(model.yaml['backbone'])][3][0] = l.out_channels

def prune_sppcspc(model, l):
    # first check if previous layers changed output and make sure input matches this change
    cin_remove = find_removed_in_channels(model, l)

    # match input dim
    l.cv1.conv = get_new_conv(l.cv1.conv, 1, cin_remove)
    l.cv2.conv = get_new_conv(l.cv2.conv, 1, cin_remove)
    l.cv3.conv = get_new_conv(l.cv3.conv, 1, l.cv1.cout_remove)
    l.cv4.conv = get_new_conv(l.cv4.conv, 1, l.cv3.cout_remove)
    cv4_cout_remove = torch.tensor(l.cv4.cout_remove)
    cv4_cout_remove = torch.cat([cv4_cout_remove + l.cv4.conv.out_channels*i for i in range(4)]).tolist()
    l.cv5.conv = get_new_conv(l.cv5.conv, 1, cv4_cout_remove)
    l.cv6.conv = get_new_conv(l.cv6.conv, 1, l.cv5.cout_remove)
    cv7_cin_remove = l.cv6.cout_remove + [x + l.cv6.conv.out_channels for x in l.cv2.cout_remove]
    l.cv7.conv = get_new_conv(l.cv7.conv, 1, cv7_cin_remove)

    # match output dim
    l.cv1.conv = get_new_conv(l.cv1.conv, 0, l.cv1.cout_remove)
    l.cv1.bn = get_new_norm(l.cv1.bn, l.cv1.cout_remove)
    l.cv2.conv = get_new_conv(l.cv2.conv, 0, l.cv2.cout_remove)
    l.cv2.bn = get_new_norm(l.cv2.bn, l.cv2.cout_remove)
    l.cv3.conv = get_new_conv(l.cv3.conv, 0, l.cv3.cout_remove)
    l.cv3.bn = get_new_norm(l.cv3.bn, l.cv3.cout_remove)
    l.cv4.conv = get_new_conv(l.cv4.conv, 0, l.cv4.cout_remove)
    l.cv4.bn = get_new_norm(l.cv4.bn, l.cv4.cout_remove)
    l.cv5.conv = get_new_conv(l.cv5.conv, 0, l.cv5.cout_remove)
    l.cv5.bn = get_new_norm(l.cv5.bn, l.cv5.cout_remove)
    l.cv6.conv = get_new_conv(l.cv6.conv, 0, l.cv6.cout_remove)
    l.cv6.bn = get_new_norm(l.cv6.bn, l.cv6.cout_remove)
    l.cv7.conv = get_new_conv(l.cv7.conv, 0, l.cv7.cout_remove)
    l.cv7.bn = get_new_norm(l.cv7.bn, l.cv7.cout_remove)

    model.yaml['head'][l.i-len(model.yaml['backbone'])][3][0] = l.cv7.conv.out_channels
    model.yaml['sppcspc'] = {
        'cv1': [l.cv1.conv.in_channels, l.cv1.conv.out_channels],
        'cv2': [l.cv2.conv.in_channels, l.cv2.conv.out_channels],
        'cv3': [l.cv3.conv.in_channels, l.cv3.conv.out_channels],
        'cv4': [l.cv4.conv.in_channels, l.cv4.conv.out_channels],
        'cv5': [l.cv5.conv.in_channels, l.cv5.conv.out_channels],
        'cv6': [l.cv6.conv.in_channels, l.cv6.conv.out_channels],
        'cv7': [l.cv7.conv.in_channels, l.cv7.conv.out_channels]
    }

    if not hasattr(l, 'remove_hist'):
        l.remove_hist = []
    l.remove_hist.append(l.cv7.cout_remove)

    l.cout_removed = l.cv7.cout_remove

def sppcspc_step(cv, y, k):
    # calculate importance score and threshold
    m_l = torch.norm(y.detach(), 1, (2,3)).mean(0)
    m_l = m_l / torch.max(m_l)
    m_l_p = k * torch.sum(m_l) / cv.conv.out_channels
    cv.cout_remove = (m_l < m_l_p).nonzero().squeeze(1).tolist()

@torch.no_grad()
def calculate_removable_channels_sppcspc(model, l, x, k):
    l_f = model.model[l.i + l.f]
    y = model.forward_till_layer(x, l_f)
    y1 = l.cv1(y)
    sppcspc_step(l.cv1, y1, k)
    y1 = l.cv3(y1)
    sppcspc_step(l.cv3, y1, k)
    y1 = l.cv4(y1)
    sppcspc_step(l.cv4, y1, k)

    y1 = l.cv5(torch.cat([y1] + [m(y1) for m in l.m], 1))
    sppcspc_step(l.cv5, y1, k)
    y1 = l.cv6(y1)
    sppcspc_step(l.cv6, y1, k)

    y2 = l.cv2(y)
    sppcspc_step(l.cv2, y2, k)

    out = l.cv7(torch.cat((y1, y2), dim=1))
    sppcspc_step(l.cv7, out, k)


@torch.no_grad()
def prune_layer(model, l, dataloader, k, num_bs, device):
    model.eval()
    model.to(device)

    cout = l.conv.out_channels
    cout_bs = torch.zeros((num_bs, cout))
    data_iter = iter(dataloader)
    m_l = torch.zeros(cout).to(device)

    l_prev = 0
    if l.f < 0:
        l_prev = l.i + l.f
    else:
        l_prev = l.f
    l_prev = model.model[l_prev]
    
    for i in range(num_bs):
        b = next(data_iter)[0]
        x = b.to(device).float() / 255.0

        if l.i == 0:
            y = l.conv(x)
        elif isinstance(l, RepConv):
            y = model.forward_till_layer(x, l)
        else:
            y = model.forward_till_layer(x, l_prev)
            y = l.conv(y)

        m_l += torch.norm(y.detach(), 1, (2,3)).mean(0)
        m_l_m = m_l / torch.max(m_l)
        m_l_p = k * torch.sum(m_l_m) / cout
        cout_remove = (m_l_m < m_l_p).nonzero().squeeze(1).tolist()
        cout_bs[i, cout_remove] = 1
    
    return cout_bs

@torch.no_grad()
def prune_step2(model, dataloader, k, num_bs, device, verbose=False):
    model.eval()
    model.to(device)

    # batch = next(data_iter)[0]
    # x = batch.to(device).float() / 255.0

    # calculate which channels to remove from each layer
    for l in tqdm(model.model):
    # for l in (model.model):
        # ignore anything but Conv layers
        if not isinstance(l, (Conv, RepConv)):
            if isinstance(l, (SPPCSPC)):
                data_iter = iter(dataloader)
                b = next(data_iter)[0]
                for i in range(3):
                    b = torch.cat((b, next(data_iter)[0]))
                calculate_removable_channels_sppcspc(model, l, x, k)
            continue

        if isinstance(l, RepConv):
            cout = l.out_channels
        else:
            cout = l.conv.out_channels

        l_prev = 0
        if l.f < 0:
            l_prev = l.i + l.f
        else:
            l_prev = l.f
        l_prev = model.model[l_prev]

        data_iter = iter(dataloader)
        m_l = torch.zeros(cout).to(device)
        for i in range(num_bs):
            b = next(data_iter)[0]
            x = b.to(device).float() / 255.0
            if l.i == 0:
                y = l.conv(x)
            elif isinstance(l, RepConv):
                y = model.forward_till_layer(x, l)
            else:
                y = model.forward_till_layer(x, l_prev)
                y = l.conv(y)
            m_l += torch.norm(y.detach(), 1, (2,3)).mean(0)
        m_l_m = m_l / torch.max(m_l)
        m_l_p = k * torch.sum(m_l_m) / cout
        cout_remove = (m_l_m < m_l_p).nonzero().squeeze(1).tolist()
        print(cout_remove)
        # calculate importance score and threshold
        # m_l = torch.norm(y.detach(), 1, (2,3)).mean(0)
        # m_l = m_l / torch.max(m_l)
        # m_l_p = k * torch.sum(m_l) / cout
        # cout_remove = (m_l < m_l_p).nonzero().squeeze(1).tolist()
        # if verbose:
        #     print(cout_remove)

        # set indices of channels to be removed
        l.cout_remove = cout_remove

    # actually remove channels and make sure input channels still match
    l_idetect = 0
    for l in model.model:
        # ignore anything but Conv layers and save Detect() layer number
        if not isinstance(l, Conv):
            if isinstance(l, IDetect):
                l_idetect = l.i
            elif isinstance(l, RepConv):
                prune_repconv(model, l)
            elif isinstance(l, SPPCSPC):
                prune_sppcspc(model, l)
            continue

        # first check if previous layers changed output and make sure input matches this change
        cin_remove = find_removed_in_channels(model, l)
        l.conv = get_new_conv(l.conv, 1, cin_remove)
        
        # remove output channels; calculated above
        l.conv = get_new_conv(l.conv, 0, l.cout_remove)
        l.bn = get_new_norm(l.bn, l.cout_remove)
        l.cout_removed = l.cout_remove
        del l.cout_remove

        if not hasattr(l, 'remove_hist'):
            l.remove_hist = []
        l.remove_hist.append(l.cout_removed)

        # update yaml
        if l.i < len(model.yaml['backbone']):
            model.yaml['backbone'][l.i][3][0] = l.conv.out_channels
        else:
            model.yaml['head'][l.i-len(model.yaml['backbone'])][3][0] = l.conv.out_channels
    
    # make sure input channels of Detect() still match
    fix_input_idetect(model, l_idetect)

    # test full model
    model.to(device)
    y = model(x)

@torch.no_grad()
def prune_step(model, img_batch, k, device, verbose=False):
    model.eval()
    model.to(device)
    x = img_batch.to(device).float() / 255.0

    # calculate which channels to remove from each layer
    for l in model.model:
        # ignore anything but Conv layers
        if not isinstance(l, (Conv, RepConv)):
            if isinstance(l, (SPPCSPC)):
                calculate_removable_channels_sppcspc(model, l, x, k)
            continue

        y = model.forward_till_layer(x, l)

        # calculate importance score and threshold
        m_l = torch.norm(y.detach(), 1, (2,3)).mean(0)
        m_l = m_l / torch.max(m_l)
        if isinstance(l, RepConv):
            m_l_p = k * torch.sum(m_l) / l.out_channels
        else:
            m_l_p = k * torch.sum(m_l) / l.conv.out_channels
        cout_remove = (m_l < m_l_p).nonzero().squeeze(1).tolist()
        if verbose:
            print(cout_remove)

        # set indices of channels to be removed
        l.cout_remove = cout_remove

    # actually remove channels and make sure input channels still match
    l_idetect = 0
    for l in model.model:
        # ignore anything but Conv layers and save Detect() layer number
        if not isinstance(l, Conv):
            if isinstance(l, IDetect):
                l_idetect = l.i
            elif isinstance(l, RepConv):
                prune_repconv(model, l)
            elif isinstance(l, SPPCSPC):
                prune_sppcspc(model, l)
            continue

        # first check if previous layers changed output and make sure input matches this change
        cin_remove = find_removed_in_channels(model, l)
        l.conv = get_new_conv(l.conv, 1, cin_remove)
        
        # remove output channels; calculated above
        l.conv = get_new_conv(l.conv, 0, l.cout_remove)
        l.bn = get_new_norm(l.bn, l.cout_remove)
        l.cout_removed = l.cout_remove
        del l.cout_remove

        if not hasattr(l, 'remove_hist'):
            l.remove_hist = []
        l.remove_hist.append(l.cout_removed)

        # update yaml
        if l.i < len(model.yaml['backbone']):
            model.yaml['backbone'][l.i][3][0] = l.conv.out_channels
        else:
            model.yaml['head'][l.i-len(model.yaml['backbone'])][3][0] = l.conv.out_channels
    
    # make sure input channels of Detect() still match
    fix_input_idetect(model, l_idetect)

    # test full model
    model.to(device)
    y = model(x)