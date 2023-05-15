import torch

import yaml
import numpy as np

from models.yolo import Model
from utils_yolo.general import fitness
from utils_yolo.test import test
from utils_yolo.load import load_data

ckpt = './tmp/training9/weights/best.pth'
data = './data/coco.yaml'
hyp = './data/hyp.scratch.tiny.yaml'

# open data info
with open(hyp) as f:
    hyp = yaml.load(f, Loader=yaml.SafeLoader)
with open(data) as f:
    data_dict = yaml.load(f, Loader=yaml.SafeLoader)

nc = int(data_dict['nc'])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

ckpt = torch.load(ckpt, map_location=device)
model = Model(ckpt['struct'], nc=nc, anchors=hyp.get('anchors'))
model.load_state_dict(ckpt['state_dict'], strict=False) 
model.to(device)
del ckpt

# test accuracy
if __name__ == '__main__':
    batch_size = 16
    num_workers = 4
    img_size = [640, 640]
    imgsz_test, dataloader, dataset, testloader, hyp, model = load_data(model, img_size, data_dict, batch_size, hyp, num_workers, device)

    results = test(
        data_dict,
        batch_size=batch_size * 2,
        imgsz=imgsz_test,
        model=model,
        dataloader=testloader,
        # compute_loss=compute_loss,
        is_coco=True,
        plots=False,
        iou_thres=0.65
    )[0]
    print('Fitness:', fitness(np.array(results).reshape(1, -1)))
