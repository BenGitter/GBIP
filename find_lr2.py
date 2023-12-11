import os
os.environ["OMP_NUM_THREADS"] = '1'

import torch

import yaml
import numpy as np
from tqdm import tqdm
from pathlib import Path
import math

from utils_yolo.load import load_model, load_gbip_model, load_data, create_optimizer
from utils_yolo.general import check_dataset, fitness, increment_path, init_seeds
from utils_yolo.test import test
from utils_yolo.loss import ComputeLossOTA
from utils_gbip.prune import prune_step2

# params
N = 1 # 30
sp = 2 # 10
k = 0.8 # (0,1) = pruning threshold factor -> 0 = no pruning, 1 = empty network

AT = False
OT = False
AG = True

augment = True
batch_size = 4
nbs = 64 # nominal batch size
accumulate = max(round(nbs / batch_size), 1)
num_workers = 1
img_size = [640, 640]
num_bs = accumulate * 4
AG_cycle = 1
warm_up = 1

data = './data/coco.vast.yaml'
hyp = './data/hyp.scratch.p5.yaml'
struct_file = './data/yolov7.yaml'
teacher_weights = './data/yolov7_training.pt'

save_dir = Path(increment_path(Path('tmp/training'), exist_ok=False))
wdir = save_dir / 'weights'
last = wdir / 'last.pth'
best = wdir / 'best.pth'
results_file = save_dir / 'results.txt'
loss_file = save_dir / 'lossesk08AG.txt'

if __name__ == "__main__":
    print('AT={}, OT={}, AG={}, k={}'.format(AT, OT, AG, k))
    # create dirs
    os.makedirs(wdir, exist_ok=True)
    init_seeds(1)

    # open data info
    with open(hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)
    with open(data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)
    with open(struct_file) as f:
        struct_file = yaml.load(f, Loader=yaml.SafeLoader)
    check_dataset(data_dict)

    # init models
    nc = int(data_dict['nc'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cuda = device.type != 'cpu'
    
    if AT or OT or AG:
        model_T = load_model(struct_file, nc, hyp.get('anchors'), teacher_weights, device).eval() # teacher model
    else:
        model_T = None
    model_S = load_model(struct_file, nc, hyp.get('anchors'), teacher_weights, device) # student model
    
    # load train+val datasets
    # data_dict['train'] = data_dict['val'] # for testing (reduces load time)
    imgsz_test, dataloader, dataset, testloader, hyp, model_S = load_data(model_S, img_size, data_dict, batch_size, hyp, num_workers, device, augment=augment)
    nb = len(dataloader)        # number of batches

    # optimizer + scaler + loss
    compute_loss = ComputeLossOTA(model_S, model_T=model_T, OT=OT, AT=AT, AG=AG)
    
    # create losses and results file and write heading
    l_file = open(loss_file, 'w')
    l_file.write(('%10s' * 13 + '\n') % ('epoch', 'gpu_mem', 'box', 'obj', 'cls', 'box_tl', 'cls_tl', 'obj_tl', 'lat', 'lag', 'total', 'lmg', 'lr'))

    # sweeping params
    lr_start = 1e-6
    lr_end = 1e0
    num_batches = 200

    hyp['lr0'] = lr_start
    gamma = (lr_end / lr_start) ** (1 / num_batches)
    optimizer, _ = create_optimizer(model_S, hyp)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    
    for g in optimizer.param_groups:
        g['lr'] = 1e-3


    lrs = []
    losses = []

    best_fitness = 0
    pruning_cycle = 0
    for epoch in range(N):
        # prune student model every sp epochs
        if epoch % sp == 0:
            # prune student model
            print('pruning')
            prune_step2(model_S, dataloader, k, num_bs, device)

            # update best_fitness + index
            best_fitness = 0
            pruning_cycle += 1
            last = wdir / 'last{}.pth'.format(pruning_cycle)
            best = wdir / 'best{}.pth'.format(pruning_cycle)
        
        mloss = torch.zeros(10, device=device)
        optimizer.zero_grad()
        ix = 0
        model_S.train()
        print(('%8s' * 13) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'box_tl', 'cls_tl', 'obj_tl', 'lat', 'lag', 'total', 'lmg', 'lr'))
        pbar = tqdm(enumerate(dataloader), total=nb)
        total_loss = 0
        for i, (imgs, targets, paths, _) in pbar:
            ni = i + nb * epoch +1
            imgs = imgs.to(device, non_blocking=True).float() / 255.0
            targets = targets.to(device)            

            # update AG this iteration?
            compute_AG = math.floor(ix/AG_cycle) % 2
            if ix < warm_up and (epoch % sp == 0):
                compute_AG = False

            # forward pass
            if AT:
                pred, att = model_S(imgs, AT=AT, attention_layers=hyp['attention_layers'])
                loss, loss_items = compute_loss(pred, targets, imgs, att, compute_AG=compute_AG)
            else:
                pred = model_S(imgs)
                loss, loss_items = compute_loss(pred, targets, imgs, compute_AG=compute_AG)

            # backprop
            loss.backward()  
            # optimize
            if ni % accumulate == 0:
                ix+=1
                losses.append(loss.item())
                lrs.append(optimizer.param_groups[0]["lr"])

                optimizer.step()
                optimizer.zero_grad()
                
                if ix == 200:
                    optimizer, _ = create_optimizer(model_S, hyp)
                    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
                elif ix > 200:
                    scheduler.step()
                # update adversarial model
                if AG:
                    loss_items[9] = compute_loss.update_AG(imgs, pred)

            if AG and not compute_AG:
                loss_items[9] = compute_loss.update_AG(imgs, pred)
            
            loss_items[7] += mloss[7] * (not compute_AG)
            loss_items[9] += mloss[9] * (compute_AG)
            
            rate = pbar.format_dict['rate']
            remaining = (pbar.total - pbar.n) / rate / 60 if rate and pbar.total else 0
            mloss = (mloss * 19 + loss_items) / 20  # update mean losses
            total_loss += loss_items[8]
            mloss[8] = total_loss / i
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%8s' * 2 + '%8.4g' * 10 + '%10.2e') % (
                '%g/%g' % (epoch, N - 1), mem, *loss_items, optimizer.param_groups[0]["lr"])
            pbar.set_description(s)

            # save losses each nominal batch
            if ni % accumulate == 0:
                s = ('%10s' * 2 + '%10.4g' * 10 + '%10.2e' + '\n') % (
                '%g/%g' % (epoch, N - 1), mem, *loss_items, optimizer.param_groups[0]["lr"])
                l_file.write(s)

            if ix == num_batches+200:
                break

        # end batch
    # end epoch