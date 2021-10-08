import matplotlib.pyplot as plt
from at.physics.dynamic_aperture import DynamicAperture
import numpy as np
from datetime import datetime
import os

def plot(DA, ax=None, file_name_save=None):
    """
    plots a figure of the acceptance for one of the defined modes

    :param Acc6D: instance of the class Acceptance6D
    :param ax: figure axes
    :param file_name_save: if given, save figure to file
    :return figure axes
    """

    if DA.mode:
        print('auto plot {m1}-{m2}'.format(m1=DA.mode[0], m2=DA.mode[1]))

        h, v, sel = DA.get_relevant_points_for_DA_computation()
        ax = plot_base(DA, h, v, sel=sel, mode=DA.mode, ax=ax, file_name_save=file_name_save)
    else:
        print('mode is None')

    return ax


def plot_base(DA, h, v, sel=None, ax=None, mode=None, file_name_save=None, font_size=10):
    """
    plots results of acceptance scan in a given 2D plane

    :param Acc6D: instance of Acceptance6D class (in pyat/at/physics/dynamic_aperture.py)
    :param h: list of x coordinates of test points
    :param v: list of y coordinates of test points
    :param sel: boolean array to mark if the specified coordinate is lost or not
    :param pl: 2 element tuple of planes default ('x', 'y')
    :param ax: figure axes
    :param file_name_save: if given, save figure to file
    :return figure axes
    """
    if ax is None:
        fig, ax = plt.subplots()

    cols = []
    for s in DA.survived:
        if s:
            cols.append('deepskyblue')
        else:
            cols.append('gainsboro')

   # get planes from mode
    pl_h, pl_v = DA.mode

    if not sel:
        sel = range(len(h))

    num_sel = [int(s) for s in sel]
    survsel = [int(s) for s in sel if s]
    if DA.verbose:
        print('{p1}-{p2} survived in {n} cases'.format(n=len(survsel), p1=pl_h, p2=pl_v))

    # apply scale factors for plot
    hs = np.array([h_ * DA.dict_units_for_plotting[pl_h][0] for h_ in h])
    vs = np.array([v_ * DA.dict_units_for_plotting[pl_v][0] for v_ in v])

    ax.scatter(hs, vs, s=20, c=cols, label='tested', facecolors='none')
    ax.plot(hs[sel], vs[sel], 'x', color='royalblue', markersize=3, label='survived')
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    try:
        cs = ax.tricontour(hs, vs, sel, linewidths=2)
        for c in cs.collections:
            c.set_edgecolor("face")

        if len(cs.allsegs) > 3:
            dat0 = cs.allsegs[-2][0]
            ax.plot(dat0[:, 0], dat0[:, 1], ':', label='limit')
        else:
            if DA.verbose:
                print('DA limit could not be computed (probably no closed contour)')
    except Exception:
        print(' limits not computed ')

    ax.set_xlabel(pl_h + ' [' + DA.dict_units_for_plotting[pl_h][1] + ']', fontsize=font_size)
    ax.set_ylabel(pl_v + ' [' + DA.dict_units_for_plotting[pl_v][1] + ']', fontsize=font_size)
    ax.set_xlim([r * DA.dict_units_for_plotting[pl_h][0] for r in DA.ranges[0]])
    if DA.grid_mode == 'radial' and DA.halfrange:
        ax.set_ylim([0, DA.ranges[1][1] * DA.dict_units_for_plotting[pl_v][0]])
    else:
        ax.set_ylim([r * DA.dict_units_for_plotting[pl_v][0] for r in DA.ranges[1]])

    ax.set_title('{m} for {t} turns\n at {ll} \n dp/p= {dpp}%, rad: {rad}'.format(
        m=DA.mode[0] + '-' + DA.mode[1], t=DA.n_turns, ll=DA.ring[0].FamName, dpp=DA.dp * 100,
        rad=DA.ring.radiation), fontsize=font_size)

    ax.legend(prop={'size': font_size})
    plt.tight_layout()

    if file_name_save:
        path, file = os.path.split(file_name_save)
        fns = file.split('.',2)

        if len(fns)>1:
            plt.savefig(path + '/' + fns[0] + '_' + DA.mode[0] + '_' + DA.mode[1] + '.' + fns[1], dpi=600)
        else:
            plt.savefig(path + '/' + fns[0] + '_' +  DA.mode[0] + '_' + DA.mode[1] , dpi=600)

    return ax



def get_border(h, v, sel):
    """
    get border from output of Acc6D.compute()


    """

    h_s = []
    v_s = []

    fig, ax = plt.subplots()
    cs = ax.tricontour(h, v, sel, linewidths=2)

    if len(cs.allsegs) > 3:
        dat0 = cs.allsegs[-2][0]
        h_s = dat0[:, 0]
        v_s = dat0[:, 1]
    else:
        print('DA limit could not be computed (probably no closed contour)')

    plt.close(fig)

    return h_s, v_s

