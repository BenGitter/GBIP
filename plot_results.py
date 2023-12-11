import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# colors
colors = ['#0C2340', '#00B8C8', '#0076C2', '#6F1D77', '#EF60A3', '#A50034', '#E03C31', '#EC6842', '#FFB81C', '#6CC24A', '#009B77']
grays = ['#888', '#9F9F9F', '#BBB', '#CFCFCF', '#EEE']

plot_one_cycle = False
plot_full_results = False
plot_pr = False
plot_act = False
plot_tiny_results = False
plot_pbs = True

# plot onecycle
if plot_one_cycle:
    import math
    def one_cycle(y1=0.0, y2=1.0, steps=100):
        # lambda function for sinusoidal ramp from y1 to y2
        return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1

    lf = one_cycle(1, .01, 90)
    x = np.arange(91)/10
    lr = np.array([lf(i*10) for i in x])

    plt.figure(figsize=(5,2))
    plt.plot(x, lr, color=colors[2])
    plt.xlim(0,9)
    plt.grid(True, 'minor', color='#DDD')
    plt.grid(True, 'major')
    plt.minorticks_on()
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate Factor')
    # plt.yscale('log')
    plt.ylim(0,1.03)
    plt.savefig("test.png", bbox_inches='tight')
    plt.show()

# plot YOLOv7 results
if plot_full_results:
    FPR_09 = [47.2, 29.9, 21.8, 17.5, 13.0]
    fitness_09 = [47.4, 41.4, 33.8, 27.5, 22.9]
    FPR_091 = [44.0, 26.2, 12.6]
    fitness_091 = [46.3, 38.9, 27.0]
    FPR_092 = [41.0, 21.2, 12.6]
    fitness_092 = [44.9, 38.3, 28.5]
    FPR_095 = [30.5, 11.3]
    fitness_095 = [41.6, 28.4]
    FPR_097 = [23.7, 6.1]
    fitness_097 = [35.0, 17.1]
    FPR_1015 = [13.1]
    fitness_1015 = [21.5]

    def plot_lines(a):
        op = 'CC'
        ms = 7
        # plot YOLOv7-tiny and vertical line
        a.plot(13.2, 39.2, marker='*', color='#0c2340', markersize=10, label='YOLOv7-tiny', linestyle='')
        a.plot([13.2, 13.2], [0, 100], color='#0c234066')
        a.plot([0, 100], [39.2, 39.2], color='#0c234066')

        c = colors[2]
        a.plot(FPR_09, fitness_09, marker='', linestyle='dashed', color=c+op)
        a.plot(FPR_09, fitness_09, marker='o', linestyle='', markersize=ms, label='k=0.9', color=c)
        c = colors[3]
        a.plot(FPR_091, fitness_091, marker='', linestyle='dashed', color=c+op)
        a.plot(FPR_091, fitness_091, marker='o', linestyle='', markersize=ms, label='k=0.91', color=c)
        c = colors[5]
        a.plot(FPR_092, fitness_092, marker='', linestyle='dashed', color=c+op)
        a.plot(FPR_092, fitness_092, marker='o', linestyle='', markersize=ms, label='k=0.92', color=c)
        c = colors[7]
        a.plot(FPR_095, fitness_095, marker='', linestyle='dashed', color=c+op)
        a.plot(FPR_095, fitness_095, marker='o', linestyle='', markersize=ms, label='k=0.95', color=c)
        # c = colors[4]
        # plt.plot(FPR_097, fitness_097, marker='', linestyle='dashed', color=c+op)
        # plt.plot(FPR_097, fitness_097, marker='o', linestyle='', markersize=9, label='k=0.97', color=c)
        c = colors[10]
        a.plot(FPR_1015, fitness_1015, marker='', linestyle='dashed', color=c+op)
        a.plot(FPR_1015, fitness_1015, marker='o', linestyle='', markersize=ms, label='k=1.015', color=c)

        

    # plt.figure(figsize=(14,6))
    # plot_lines()
    # plt.grid(True, 'minor', color='#DDD')
    # plt.grid(True, 'major')
    # plt.minorticks_on()
    # plt.xlim(0, 50)
    # plt.ylim(15, 50)
    # plt.xlabel('FLOPs Pruning Ratio (FPR)')
    # plt.ylabel('fitness')
    # # plt.gca().xaxis.set_major_locator(plt.MultipleLocator(.1))
    # plt.legend()
    # plt.show()
    
    # plt.figure(figsize=(14,6))
    # plot_lines()
    # plt.grid(True, 'minor', color='#DDD')
    # plt.grid(True, 'major')
    # plt.minorticks_on()
    # plt.xlim(10, 16)
    # plt.ylim(20, 30)
    # plt.xlabel('FLOPs Pruning Ratio (FPR)')
    # plt.ylabel('fitness')
    # # plt.gca().xaxis.set_major_locator(plt.MultipleLocator(.1))
    # plt.legend()
    # plt.show()

    f, (a0, a1) = plt.subplots(1, 2, figsize=(14,4), width_ratios=[3, 1])
    plot_lines(a0)
    # a0.plot(13.2, 39.20, marker='*', color='#0c2340', markersize=10, label='YOLOv7-tiny', linestyle='')
    # a0.plot([13.2, 13.2], [0, 100], color='#0c234099', linestyle='dashed', zorder=0)
    a0.add_patch(Rectangle((10.5, 20.5), 4, 9, edgecolor='black', facecolor='none', lw=1))

    a0.grid(True, 'minor', color='#DDD')
    a0.grid(True, 'major')
    a0.minorticks_on()
    a0.set_axisbelow(True)
    a0.set_xlim(10, 50)
    a0.set_ylim(20, 50)
    a0.set_xlabel('FLOPs Pruning Ratio (FPR)')
    a0.set_ylabel('Fitness')
    a0.legend(loc='lower right')
    

    plot_lines(a1)
    a1.add_patch(Rectangle((11, 21), 3, 8, edgecolor='black', facecolor='none', lw=3))
    a1.grid(True, 'minor', color='#DDD')
    a1.grid(True, 'major')
    a1.minorticks_on()
    a1.set_axisbelow(True)
    a1.set_xlim(11, 14)
    a1.set_ylim(21, 29)
    a1.set_xlabel('FLOPs Pruning Ratio (FPR)')
    a1.set_ylabel('Fitness')
    plt.savefig("test.png", bbox_inches='tight')
    plt.show()

# plot precision-recall curve
if plot_pr:
    p = [1.0, 1.0, 0.5, 0.67, 0.4, 0.5, 0.43]
    p_interp = [1.0, 1.0, 0.67, 0.67, 0.5, 0.5, 0.5]
    r = [0.0, 0.33, 0.33, .67, .67, 1.0, 1.0]
    
    plt.figure(figsize=((14,4)))
    plt.plot(r, p, label=r'$p(r)$', color=colors[9])
    plt.plot(r, p_interp, label=r'$p_{interp}(r)$', color=colors[7])
    plt.fill_between(r, p_interp, color=colors[7]+'11', hatch='/', edgecolor=colors[7]+'33')
    plt.xlim(0, 1.02)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.xlabel(r'recall $r$')
    plt.ylabel(r'precision $p$')
    plt.grid(True, 'minor', color='#DDD')
    plt.grid(True, 'major')
    # plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
    # plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(.1))
    plt.minorticks_on()
    plt.savefig("test.png", bbox_inches='tight')
    plt.show()

# plot activation functions
if plot_act:
    import torch
    x = torch.linspace(-6.1, 6.1, 101)
    y_relu = torch.nn.functional.relu(x)
    y_sigmoid = torch.sigmoid(x)
    y_silu = torch.nn.functional.silu(x)

    # plt.figure(figsize=((14, 6)))
    fig, axes = plt.subplots(1, 3, figsize=(14,2))

    axes[0].plot(x, y_relu, color=colors[2])
    axes[1].plot(x, y_silu, color=colors[9])
    axes[2].plot(x, y_sigmoid, color=colors[7])

    axes[0].grid(True, 'minor', color='#DDD')
    axes[0].grid(True, 'major')
    axes[0].minorticks_on()
    axes[1].grid(True, 'minor', color='#DDD')
    axes[1].grid(True, 'major')
    axes[1].minorticks_on()
    axes[2].grid(True, 'minor', color='#DDD')
    axes[2].grid(True, 'major')
    axes[2].minorticks_on()
    # plt.xlim(0,1.02)
    axes[0].set_xlabel('x')
    axes[1].set_xlabel('x')
    axes[2].set_xlabel('x')
    axes[0].set_ylabel('f(x)')
    axes[1].set_ylabel('f(x)')
    axes[2].set_ylabel('f(x)')

    axes[0].legend(['ReLU'])
    axes[1].legend(['SiLU'])
    axes[2].legend(['sigmoid'])

    axes[0].set_xlim(-6, 6)
    axes[1].set_xlim(-6, 6)
    axes[2].set_xlim(-6, 6)
    # axes[0].set_title(['ReLU'])
    # axes[1].set_title(['SiLU'])
    # axes[2].set_title(['sigmoid'])
    # plt.gca().xaxis.set_major_locator(plt.MultipleLocator(.1))
    # plt.legend(['ReLU', 'SiLU', 'sigmoid'])
    plt.savefig("test.png", bbox_inches='tight')
    plt.show()

# plot YOLOv7-tiny results
if plot_tiny_results:
    FPR_GBIP = np.array([0.99816024, 0.997684066, 0.995923668, 0.952296093, 0.941387396, 0.935673316, 0.904794199, 0.877075142, 0.868056708, 0.803102341, 0.724750189, 0.68637495, 0.652811948, 0.491850943, 0.422257494])
    fitness_GBIP = np.array([0.3892, 0.3893, 0.3900, 0.3850, 0.3840, 0.3843, 0.3795, 0.3764, 0.3759, 0.3673, 0.3541, 0.3479, 0.3392, 0.2984, 0.2801])
    max_idx_GBIP = np.array([2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

    FPR_KSE_theory = np.array([0.674, 0.557, 0.463, 0.396, 0.547, 0.404, 0.323, 0.274])
    FPR_KSE_actual = np.array([0.929, 0.967, 0.979, 0.982, 0.929, 0.967, 0.979, 0.982])
    fitness_KSE = np.array([0.3688, 0.3678, 0.3602, 0.3479, 0.3599, 0.3541, 0.3407, 0.3175])
    max_idx_KSE = np.array([0, 1, 2, 5, 6, 7])

    plt.figure(figsize=((14, 6)))

    # plot GBIP
    c1 = colors[2]
    c2 = colors[2] + '44'
    plt.plot(100*FPR_GBIP, 100*fitness_GBIP, marker='o', color=c1, linestyle='', markersize=7)
    plt.plot(100*FPR_GBIP[max_idx_GBIP], 100*fitness_GBIP[max_idx_GBIP], marker='o', color=c2, linestyle='dashed')

    # plot KSE (actual)
    c1 = colors[9]
    c2 = colors[9] + '44'
    plt.plot(100*FPR_KSE_actual, 100*fitness_KSE, marker='o', color=c1, linestyle='', markersize=7)

    # plot KSE (theory)
    c1 = colors[7]
    c2 = colors[7] + '44'
    plt.plot(100*FPR_KSE_theory, 100*fitness_KSE, marker='o', color=c1, linestyle='', markersize=7)
    plt.plot(100*FPR_KSE_theory[max_idx_KSE], 100*fitness_KSE[max_idx_KSE], marker='o', color=c2, linestyle='dashed')

    # plt.plot(1, 0.3920, marker='*', color='#000', markersize=10)

    plt.grid(True, 'minor', color='#DDD')
    plt.grid(True, 'major')
    plt.minorticks_on()
    plt.xlim(20,102)
    plt.xlabel('FLOPs Pruning Ratio (FPR)')
    plt.ylabel('Fitness')
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(10))
    plt.legend(['GBIP', '_', 'KSE (actual)', 'KSE (theoretical)'])
    plt.savefig("test.png", bbox_inches='tight')
    plt.show()

# plot pruning batch size
if plot_pbs:
    # cout taken at k=0.9
    # idx = [11, 63]
    idx = [11, 63, 24, 101]
    cout0 = torch.load('cout_l{}_k08.pt'.format(idx[0]))
    cout1 = torch.load('cout_l{}_k08.pt'.format(idx[1]))
    cout2 = torch.load('cout_l{}_k08.pt'.format(idx[2]))
    cout3 = torch.load('cout_l{}_k08.pt'.format(idx[3]))

    from matplotlib import colors as clrs
    cmap_custom = clrs.ListedColormap(['black', colors[7]])

    fig, axes = plt.subplots(2, 2, figsize=(14, 5))
    
    y = np.arange(cout0.shape[1]+1)
    x = np.linspace(0, cout0.shape[0]*2, cout0.shape[0]+1)
    axes[0, 0].pcolormesh(x, y, np.array(cout0.t()), cmap=cmap_custom)

    y = np.arange(cout1.shape[1]+1)
    x = np.linspace(0, cout1.shape[0]*2, cout1.shape[0]+1)
    axes[0, 1].pcolormesh(x, y, np.array(cout1.t()), cmap=cmap_custom)

    y = np.arange(cout2.shape[1]+1)
    x = np.linspace(0, cout2.shape[0]*2, cout2.shape[0]+1)
    axes[1, 0].pcolormesh(x, y, np.array(cout2.t()), cmap=cmap_custom)

    y = np.arange(cout3.shape[1]+1)
    x = np.linspace(0, cout3.shape[0]*2, cout3.shape[0]+1)
    axes[1, 1].pcolormesh(x, y, np.array(cout3.t()), cmap=cmap_custom)
    # axes[0, 0].imshow(np.array(cout0.t()), cmap='hot', interpolation='nearest', extent=ext0, aspect=1)
    
    axes[0, 0].set_xscale('symlog', base=2)
    axes[0, 1].set_xscale('symlog', base=2)
    axes[1, 0].set_xscale('symlog', base=2)
    axes[1, 1].set_xscale('symlog', base=2)
    axes[0, 0].set_xlim(4, 2048)
    axes[0, 1].set_xlim(4, 2048)
    axes[1, 0].set_xlim(4, 2048)
    axes[1, 1].set_xlim(4, 2048)

    axes[0, 0].set_xlabel('Number of Training Samples')
    axes[0, 1].set_xlabel('Number of Training Samples')
    axes[1, 0].set_xlabel('Number of Training Samples')
    axes[1, 1].set_xlabel('Number of Training Samples')
    axes[0, 0].set_ylabel('Output Channels')
    axes[0, 0].set_ylabel('Output Channels')
    axes[0, 1].set_ylabel('Output Channels')
    axes[1, 0].set_ylabel('Output Channels')
    axes[1, 1].set_ylabel('Output Channels')
    axes[0, 0].set_title('Layer {}'.format(idx[0]))
    axes[0, 1].set_title('Layer {}'.format(idx[1]))
    axes[1, 0].set_title('Layer {}'.format(idx[2]))
    axes[1, 1].set_title('Layer {}'.format(idx[3]))

    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.savefig("test.png", bbox_inches='tight')
    plt.show()
