import sys
import os
os.environ["OMP_NUM_THREADS"] = '1'

import torch
import numpy as np
import yaml
from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count

from utils_yolo.load import load_gbip_model
from utils_yolo.general import check_dataset, fitness
from utils_yolo.datasets import create_dataloader
from utils_yolo.test import test
from utils_yolo.torch_utils import model_info

if __name__ == '__main__':
    exp_file = '../saved_models/GBIP/experiments_full/update_lr/k1015/ignore_l1/best{}.pth'
    num_exp = 1

    # params
    batch_size = 8
    num_workers = 2
    hyp = './data/hyp.scratch.p5.yaml'
    data = './data/coco.yaml'
    
    # open data info
    with open(hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)
    with open(data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)
    check_dataset(data_dict)

    nc = int(data_dict['nc'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load data
    gs = 32
    imgsz = 640
    testloader = create_dataloader(data_dict['val'], imgsz, batch_size, gs, hyp=hyp, 
                                    rect=True, workers=num_workers, pad=0.5)[0]
    
    for i in range(1, num_exp+1):
        weights = exp_file.format(i)

        print('========================================================================================')
        print('Evaluating {}'.format(weights))

        # load model
        model = load_gbip_model(weights, nc, device)

        num_params = sum(p.numel() for p in model.parameters())
        num_flops, params, ret_dict = model_info(model, verbose=False, report_missing=False)

        # evaluate on MS COCO validation set
        results_all, _, _, eval_all = test(
            data_dict,
            batch_size=batch_size,
            imgsz=imgsz,
            model=model,
            dataloader=testloader,
            is_coco=True,
            plots=False,
            iou_thres=0.65,
            save_json=True
        )

        print('Fitness\t\t', fitness(np.array(results_all).reshape(1, -1))[0])
        print('#params\t\t', num_params / 1e6)
        print('#FLOPs\t\t', num_flops)
        print('Finished evaluating {}'.format(weights))