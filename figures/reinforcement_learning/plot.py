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

    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(1, 1, 1)

    x = [i for i in range(1, 501)]

    ax.plot(x, losses[0], color='blue', ls='solid', label='Constant')
    ax.plot(x, losses[1], color='red', ls='solid', label='Gravity')
    ax.plot(x, losses[2], color='green', ls='solid', label='Random')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Reward')

    plt.legend()
    plt.savefig('rl_training.png', format='png', bbox_inches='tight', dpi=400)
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

    i = 0
    losses = []
    for p in paths:
        losses.append(grab_loss(p))
    plot_training(losses)
