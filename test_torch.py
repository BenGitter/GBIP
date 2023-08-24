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

if __name__ == '__main__':
    # assert len(sys.argv) > 1, 'Provide filename as cmd argument'
    if len(sys.argv) == 1:
        sys.argv.append('yolov7-tiny.pth')
    
    # params
    batch_size = 16
    num_workers = 4
    hyp = './data/hyp.scratch.tiny.yaml'
    data = './data/coco.yaml'
    weights = sys.argv[1]
    # weights = '../saved_models/GBIP/ablation/OT2.pth'

    # open data info
    with open(hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)
    with open(data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)
    check_dataset(data_dict)

    # load model
    nc = int(data_dict['nc'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = load_gbip_model(weights, nc, device)

    # FLOPs + params
    num_params = sum(p.numel() for p in model.parameters())
    num_flops = 0

    
    # inputs = torch.zeros((1, 3, 640, 640)).to(device)
    # flops = FlopCountAnalysis(model, inputs)
    # table = flop_count_table(flops)
    # params = parameter_count(model)['']
    # print(table)

    # exit()
    # print('---   torch   ---')
    # print('#params:', num_params)

    # print('--- fvcore.nn ---')
    # print(f'#params: {params} \t FLOPs: {flops.total()}')

    from utils_yolo.torch_utils import model_info
    flops, params, ret_dict = model_info(model, verbose=False, report_missing=False)
    print(flops, params)
    
    import pprint
    pp = pprint.PrettyPrinter(sort_dicts=False)
    # pp.pprint(ret_dict['model'][2])
    # print(ret_dict['model'][2])

    bn_total = 0
    for layer in ret_dict['model'][2].items():
        if 'bn' in layer[1][2]:
            bn_total += layer[1][2]['bn'][0]

    print(bn_total, ret_dict['model'][0], bn_total / ret_dict['model'][0] * 100)

    exit()
    # load data
    gs = max(int(model.stride.max()), 32)
    imgsz = 640
    testloader = create_dataloader(data_dict['val'], imgsz, batch_size, gs, hyp=hyp, 
                                    rect=True, workers=num_workers, pad=0.5)[0]

    # evaluate on MS COCO validation set
    print('=========================================================')
    print('Running test() for all categories...')
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

    # evaluate on MS COCO validation set
    print('=========================================================')
    print('Running test() for "person" category...')
    eval_p = test(
        data_dict,
        batch_size=batch_size,
        imgsz=imgsz,
        model=model,
        dataloader=testloader,
        is_coco=True,
        plots=False,
        iou_thres=0.65,
        save_json=True,
        coco_only_person=True
    )[3]

    print('=========================================================')
    print('AP\t\t', eval_all[0])
    print('AP50\t\t', eval_all[1])
    print('AP_person\t', eval_p[0])
    print('AP_person_L\t', eval_p[5])
    print('Fitness\t\t', fitness(np.array(results_all).reshape(1, -1))[0])
    print('#params\t\t', num_params)
    print('#FLOPs\t\t', num_flops)
    # pretty print?