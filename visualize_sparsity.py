import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt

from utils_yolo.load import load_model, load_gbip_model
from utils_yolo.general import check_dataset
from models.common import Conv, SPPCSPC


weights = '../saved_models/GBIP/experiments_full/update_lr/k09/best2.pth'

data = './data/coco.yaml'
hyp = './data/hyp.scratch.p5.yaml'
struct_file = './data/yolov7.yaml'
teacher_weights = './data/yolov7_training.pt'

save_PR = 0
plots = 1
testing = 0

colors = ['#0C2340', '#00B8C8', '#0076C2', '#6F1D77', '#EF60A3', '#A50034', '#E03C31', '#EC6842', '#FFB81C', '#6CC24A', '#009B77']
grays = ['#888', '#9F9F9F', '#BBB', '#CFCFCF', '#EEE']
opacities = ['FF', 'EE', 'DD', 'CC', 'BB', 'AA', '99', '88', '77', '66', '55']

def add_sppcspc(l):
    total = 0
    for i in range(1, 8):
        c = getattr(l, 'cv'+str(i))
        total += c.conv.out_channels * c.conv.in_channels
    return total

if __name__ == '__main__':
    if save_PR:
        # open data info
        with open(hyp) as f:
            hyp = yaml.load(f, Loader=yaml.SafeLoader)
        with open(data) as f:
            data_dict = yaml.load(f, Loader=yaml.SafeLoader)
        with open(struct_file) as f:
            struct_file = yaml.load(f, Loader=yaml.SafeLoader)
        check_dataset(data_dict)

        nc = int(data_dict['nc'])
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model_T = load_model(struct_file, nc, hyp.get('anchors'), teacher_weights, device).eval()

        params_T = []
        params_S = []
        params_idx = []
        for l in model_T.model:
            if isinstance(l, Conv):
                params_T.append(l.conv.out_channels * l.conv.in_channels)
                params_idx.append(l.i)
            elif isinstance(l, SPPCSPC):
                params_T.append(add_sppcspc(l))
                params_idx.append(l.i)

        # params_T = [l.conv.out_channels * l.conv.in_channels for l in model_T.model if isinstance(l, Conv)]
        # params_T.append(add_sppcspc(model_T.model[51]))
        

        # params_idx = [l.i for l in model_T.model if isinstance(l, Conv)]
        # params_idx.append(51)

        torch.save(params_idx, 'PR_idx.pt')
        file = '../saved_models/GBIP/experiments_full/update_lr/k1015/best{}.pth'
        params_S = []
        for i in range(1, 2):
            weights = file.format(i)
            model_S = load_gbip_model(weights, nc, device).eval()
            params_S.append([])
            for l in model_S.model:
                if isinstance(l, Conv):
                    params_S[i-1].append(l.conv.out_channels * l.conv.in_channels)
                elif isinstance(l, SPPCSPC):
                    params_S[i-1].append(add_sppcspc(l))
            # params_S.append([l.conv.out_channels * l.conv.in_channels for l in model_S.model if isinstance(l, Conv)])
            # params_S[i-1].append(add_sppcspc(model_S.model[51]))

        PR = torch.tensor(np.array(params_S) / np.array(params_T) * 100)
        # torch.save(PR, 'PR_1015.pt')

    if testing:
        PR_09 = torch.load('PR_09.pt')
        PR_091 = torch.load('PR_091.pt')
        PR_092 = torch.load('PR_092.pt')
        PR_095 = torch.load('PR_095.pt')
        PR_097 = torch.load('PR_097.pt')
        PR_1015 = torch.load('PR_1015.pt')
        PR_idx = torch.load('PR_idx.pt')

        plt.figure(figsize=(14,4))
        plt.plot(PR_idx, PR_09[-1, :])
        plt.plot(PR_idx, PR_092[-1, :])
        plt.plot(PR_idx, PR_097[-1, :])
        plt.show()

    if plots:
        PR_09 = torch.load('PR_09.pt')
        PR_091 = torch.load('PR_091.pt')
        PR_092 = torch.load('PR_092.pt')
        PR_095 = torch.load('PR_095.pt')
        PR_097 = torch.load('PR_097.pt')
        PR_1015 = torch.load('PR_1015.pt')
        PR_idx = torch.load('PR_idx.pt')

        plt.figure(figsize=(14,4))
        for i in range(0, PR_09.shape[0]):
            c = colors[2] + opacities[8 - 2*i]
            plt.plot(PR_idx, PR_09[i, :], color=c)

        plt.grid(True, 'minor', color='#DDD')
        plt.grid(True, 'major')
        plt.minorticks_on()
        plt.legend(['Step 1', 'Step 2', 'Step 3', 'Step 4', 'Step 5'])
        plt.xlabel('Layer Index')
        plt.ylabel('Pruning Ratio')
        plt.xlim(0, 101)
        plt.ylim(0, 100)
        plt.savefig('test.png', bbox_inches='tight')
        plt.show()


        # plt.figure(figsize=(14,4))
        # for i in range(0, PR_091.shape[1]):
        #     c = colors[2] + opacities[8 - 2*i]
        #     plt.plot(PR_091[:, i], color=c)

        # plt.grid(True, 'minor', color='#DDD')
        # plt.grid(True, 'major')
        # plt.minorticks_on()
        # plt.show()


        fig, axes = plt.subplots(1, 2, figsize=(14,4))  
        axes[0].plot(PR_idx, PR_09[0, :], color=colors[0]+opacities[0], label='k=0.9')
        axes[0].plot(PR_idx, PR_092[0, :], color=colors[2]+opacities[0], label='k=0.92')
        axes[0].plot(PR_idx, PR_095[0, :], color=colors[1]+opacities[0], label='k=0.95')
        axes[0].plot(PR_idx, PR_1015[0, :], color=colors[9]+opacities[0], label='k=1.015')

        axes[1].plot(PR_idx, PR_09[-1, :], color=colors[0]+opacities[0], label='k=0.9')
        axes[1].plot(PR_idx, PR_092[-1, :], color=colors[2]+opacities[0], label='k=0.92')
        axes[1].plot(PR_idx, PR_095[-1, :], color=colors[1]+opacities[0], label='k=0.95')
        axes[1].plot(PR_idx, PR_1015[-1, :], color=colors[9]+opacities[0], label='k=1.015')

        axes[0].grid(True, 'minor', color='#DDD')
        axes[0].grid(True, 'major')
        axes[0].minorticks_on()
        axes[0].legend()
        axes[0].set_xlim(0, 101)
        axes[0].set_ylim(0, 100)
        axes[0].set_xlabel('Layer Index')
        axes[0].set_ylabel('Pruning Ratio')

        axes[1].grid(True, 'minor', color='#DDD')
        axes[1].grid(True, 'major')
        axes[1].minorticks_on()
        axes[1].legend()
        axes[1].set_xlim(0, 101)
        axes[1].set_ylim(0, 100)
        axes[1].set_xlabel('Layer Index')
        axes[1].set_ylabel('Pruning Ratio')

        fig.savefig('test.png', bbox_inches='tight')
        plt.show()

        

        # plt.figure(figsize=(14,4))
        # plt.plot(PR_09[:, -1], color=colors[0]+opacities[0], label='k=0.9')
        # # plt.plot(PR_091[:, -1], color=colors[1]+opacities[0], label='k=0.91')
        # plt.plot(PR_092[:, -1], color=colors[2]+opacities[0], label='k=0.92')
        # plt.plot(PR_095[:, -1], color=colors[3]+opacities[0], label='k=0.95')
        # # plt.plot(PR_097[:, -1], color=colors[4]+opacities[0], label='k=0.97')
        # plt.plot(PR_1015[:, -1], color=colors[5]+opacities[0], label='k=1.015')

        # plt.grid(True, 'minor', color='#DDD')
        # plt.grid(True, 'major')
        # plt.minorticks_on()
        # plt.legend()
        # plt.show()


        # plt.figure(figsize=(14,4))
        # plt.plot(PR_09[:, 0], color=colors[0]+opacities[0], label='k=0.9')
        # # plt.plot(PR_091[:, 0], color=colors[1]+opacities[0], label='k=0.91')
        # plt.plot(PR_092[:, 0], color=colors[2]+opacities[0], label='k=0.92')
        # plt.plot(PR_095[:, 0], color=colors[3]+opacities[0], label='k=0.95')
        # # plt.plot(PR_097[:, 0], color=colors[4]+opacities[0], label='k=0.97')
        # plt.plot(PR_1015[:, 0], color=colors[5]+opacities[0], label='k=1.015')

        # plt.grid(True, 'minor', color='#DDD')
        # plt.grid(True, 'major')
        # plt.minorticks_on()
        # plt.legend()
        # plt.show()