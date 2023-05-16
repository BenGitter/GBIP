import os
os.environ["OMP_NUM_THREADS"] = '1'

import torch

import yaml
import numpy as np
from tqdm import tqdm
from pathlib import Path

from utils_yolo.load import load_model, load_gbip_model, load_data, create_optimizer
from utils_yolo.general import check_dataset, fitness, increment_path, init_seeds
from utils_yolo.test import test
from utils_yolo.loss2 import ComputeLossOTA
from utils_gbip.prune import prune_step

# params
N = 2 # 30
sp = 10 # 10
k = 0.2 # (0,1) = pruning threshold factor -> 0 = no pruning, 1 = empty network

batch_size = 16
nbs = 64 # nominal batch size
accumulate = max(round(nbs / batch_size), 1)
num_workers = 4
img_size = [640, 640]

lr = 5e-3
lr_gamma = 1

data = './data/coco.yaml'
hyp = './data/hyp.scratch.tiny.yaml'
struct_file = './data/yolov7_tiny_struct.yaml'
teacher_weights = './data/yolov7-tiny.pt'

save_dir = Path(increment_path(Path('tmp/training'), exist_ok=False))
wdir = save_dir / 'weights'
last = wdir / 'last.pth'
best = wdir / 'best.pth'
results_file = save_dir / 'results.txt'

if __name__ == "__main__":
    # create dirs
    os.makedirs(wdir, exist_ok=True)
    init_seeds(1)

    # open data info
    with open(hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)
        hyp['lr0'] = lr
        hyp['lr_gamma'] = lr_gamma
    with open(data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)
    with open(struct_file) as f:
        struct_file = yaml.load(f, Loader=yaml.SafeLoader)
    check_dataset(data_dict)

    # init models
    nc = int(data_dict['nc'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cuda = device.type != 'cpu'
    model_T = load_model(struct_file, nc, hyp.get('anchors'), teacher_weights, device).eval() # teacher model
    model_S = load_model(struct_file, nc, hyp.get('anchors'), teacher_weights, device) # student model
    # model_G = # adversarial model

    # load train+val datasets
    # data_dict['train'] = data_dict['val'] # for testing (reduces load time)
    imgsz_test, dataloader, dataset, testloader, hyp, model_S = load_data(model_S, img_size, data_dict, batch_size, hyp, num_workers, device)
    nb = len(dataloader)        # number of batches

    # optimizer + scaler + loss
    optimizer, scheduler = create_optimizer(model_S, hyp)
    compute_loss = ComputeLossOTA(model_S, model_T=model_T)

    best_fitness = 0
    for epoch in range(N):
        # prune student model every sp epochs
        if epoch % sp == 0:
            # create one large batch
            data_iter = iter(dataloader)
            batch = next(data_iter)[0]
            # for i in range(3):
            #     batch = torch.cat((batch, next(data_iter)[0]))
            # prune student model
            print('pruning')
            prune_step(model_S, batch, k, device)
            optimizer, scheduler = create_optimizer(model_S, hyp)
            del data_iter, batch
        
        mloss = torch.zeros(7, device=device)
        optimizer.zero_grad()
        ix = 0
        model_S.train()
        print(('%10s' * 10) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'box_tl', 'kl_cls', 'kl_obj', 'total', 'labels'))
        pbar = tqdm(enumerate(dataloader), total=nb)
        for i, (imgs, targets, paths, _) in pbar:
            ni = i + nb * epoch +1
            imgs = imgs.to(device, non_blocking=True).float() / 255.0
            targets = targets.to(device)

            # update model_G (while keeping model_S fixed)
            

            # update model_S (while keeping model_G fixed)
            # forward pass
            pred = model_S(imgs)
            loss, loss_items = compute_loss(pred, targets, imgs)

            # backprop
            loss.backward()

            # optimize
            if ni % accumulate == 0:
                ix+=1
                optimizer.step()
                optimizer.zero_grad()
            
             # print
            rate = pbar.format_dict['rate']
            remaining = (pbar.total - pbar.n) / rate / 60 if rate and pbar.total else 0
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.4g' * 8) % (
                '%g/%g' % (epoch, N - 1), mem, *mloss, targets.shape[0])
            pbar.set_description(s)

            if ix == 10:
                break

        # end batch
        scheduler.step()

        # run validation at end of each epoch
        results, maps = test(
            data_dict,
            batch_size=batch_size * 2,
            imgsz=imgsz_test,
            model=model_S,
            dataloader=testloader,
            # compute_loss=compute_loss,
            is_coco=True,
            plots=False,
            iou_thres=0.65
        )[0:2]
        print('Person mAP:', maps[0])
        fi = fitness(np.array(results).reshape(1, -1))
        if fi > best_fitness:
            best_fitness = fi
            print('Found higher fitness:', best_fitness)
        else:
            print('Fitness not higher:', fi)

        # save last + best
        torch.save({
            'state_dict': model_S.state_dict(),
            'struct': model_S.yaml
        }, last)
        if best_fitness == fi:
            torch.save({
                'state_dict': model_S.state_dict(),
                'struct': model_S.yaml
            }, best)
    # end epoch

    # test best.pth
    results, _, _, stats = test(
        data_dict,
        batch_size=batch_size * 2,
        imgsz=imgsz_test,
        conf_thres=0.001,
        iou_thres=0.65,
        model=load_gbip_model(best, nc, hyp.get('anchors'), device),
        dataloader=testloader,
        save_dir=save_dir,
        save_json=True,
        plots=True,
        is_coco=True
    )
    torch.cuda.empty_cache()


# Effect of using Attention Transfer (AT), Output Transfer (OT) and/or Adversarial Game (AG):
#        ResNet-56/CIFAR-100    ResNet-18/ImageNet
# 1. AT:    +.12                    +.14    (3)
# 2. OT:    +.15                    +.42    (1)
# 3. AG:    +.13                    +.25    (2)

# TODO:
# [X] Implement pruning step
# [-] Implement Teacher + Student training (most effective)
# [-] Try Attention Transfer (minor effect, but should be relatively easy)
# [-] Try Adversarial Game (larger effect, but probably harder)