import sys
import tensorflow as tf
import matplotlib
from matplotlib import pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

def plot_training(losses):
    #plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')

    fig = plt.figure(figsize=(6, 3))
    ax = fig.add_subplot(1, 1, 1)

    x = [i for i in range(1, len(losses[0])+1)]
    print(losses[0])
    print(len(losses[0]))

    ax.plot(x, losses[0], color='blue', ls='dashed', label='Conv. Avg.')
    ax.plot(x, losses[2], color='green', ls='dashed', label='Conv. Cycle')
    ax.plot(x, losses[4], color='brown', ls='dashed', label='Conv. Gravity')
    ax.plot(x, losses[1], color='red', ls='solid', label='F.C. Avg.')
    ax.plot(x, losses[3], color='yellow', ls='solid', label='F.C. Cycle')
    ax.plot(x, losses[5], color='purple', ls='solid', label='F.C. Gravity')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Squared Error')

    plt.legend()
    plt.savefig('su_training.png', format='png', bbox_inches='tight', dpi=400)
    plt.show()
    return


def grab_loss(path):
    loss = []
    for e in tf.train.summary_iterator(path):
        for v in e.summary.value:
            loss.append(v.simple_value)
    return loss


if __name__ == '__main__':
    paths = ['./constant-fc-1547522469.31/events.out.tfevents.1547522502.visiongpu5',
             './gravity-cycle-fc-1547528314.94/events.out.tfevents.1547528393.visiongpu5',
             './gravity-random-fc-1547522090.95/events.out.tfevents.1547522125.visiongpu5']
    paths = ['./gravity-avg_conv1543702766.67/events.out.tfevents.1543702766.visiongpu5',
            './gravity-avg_fc1543702879.65/events.out.tfevents.1543702879.visiongpu5',
            './gravity-cycle_conv1543702625.3/events.out.tfevents.1543702625.visiongpu5',
            './gravity-cycle_fc1543702905.12/events.out.tfevents.1543702905.visiongpu5',
            './gravity-random_conv1543702661.69/events.out.tfevents.1543702661.visiongpu5',
            './gravity-random_fc1543703055.23/events.out.tfevents.1543703055.visiongpu5']
    i = 0
    losses = []
    for p in paths:
        losses.append(grab_loss(p))
    for i in range(len(losses)):
        losses[i] = losses[i][::2]
    plot_training(losses)
