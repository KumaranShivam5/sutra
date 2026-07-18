import numpy as np 
from matplotlib import pyplot as plt

import importlib
import astrools

# Force a reload of the module every time the script runs
importlib.reload(astrools)

from astrools.image import Image , make_wcs_good 
from astrools.scale import scale_fits_image

from mpl_toolkits.axes_grid1 import make_axes_locatable

from astropy.wcs import WCS



import copy
def plot_cd(cd, skeleton = None, axlabels = True, *args, **kwargs):

    tmp = copy.deepcopy(cd.data)
    tmp[tmp<1e10] = np.nan
    tmp[tmp>1e28] = np.nan
    vmin = np.nanpercentile(tmp, 10)
    vmax = np.nanpercentile(tmp, 99.9)

    fig, ax = plt.subplots(1,1 ,  subplot_kw={"projection": WCS(cd.header)},  *args, **kwargs)

    im = ax.imshow(cd.data, cmap = 'YlOrBr_r', vmin = vmin , vmax = vmax, norm = 'log',)
    ax = make_wcs_good(ax)
    if skeleton is not None:
        ax.contour(skeleton, linewidths = 0.5, colors = 'k', levels = 1)
    ax.set_facecolor('k')
    fig.set_facecolor("#00000000") 
    fig.patch.set_alpha(0)
    return fig



def plot_mask(mask, header):


    fig, ax = plt.subplots(1,1 ,  subplot_kw={"projection": WCS(header)})
    ax.imshow(mask)
    im = ax.contourf(mask, levels = 1 , colors = ['k', 'brown'])
    ax = make_wcs_good(ax)
    
    return fig




def plot_prob(prob_map , header):

    fig, ax = plt.subplots(1,1 , )
    im = ax.imshow(prob_map.T.T, cmap='afmhot',  vmax = 0.8, origin='lower')
    # im = ax.contourf(mask, levels = 1 , colors = ['k', 'w'])
    # ax = make_wcs_good(ax)
    ax.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size = "5%", pad = 0.05)
    cbar = fig.colorbar(im,cax = cax )
    # cbar = fig.colorbar(im, )
    cax.set_xticks([])
    # cax.set_yticks([])
    cax.xaxis.set_visible(False)
    cax.yaxis.set_ticks_position('right')
    cax.yaxis.set_label_position('right')
    cbar.set_label("Filament Crest Probability", fontsize = 14)
    cax.tick_params(axis='y', labelsize = 12)
    return fig


# @st.cache_resource


def plot_radials(radprof):
    radprof.img[radprof.img<1e10] = np.nan
    vmin = np.nanpercentile(radprof.img, 10)
    vmax = np.nanpercentile(radprof.img, 99)
    plt.close()
    fig = plt.figure()
    ax = plt.subplot(111,)
    ax.imshow(radprof.img , cmap = 'YlOrBr_r', norm = 'log', vmin = vmin, vmax = vmax, origin='lower')
    ax.contour(radprof.skel, colors = 'k', linewidths = 0.2, alpha = 1, zorder=2)
    for p in radprof.prof_dict:
        l ,h = p['low'] , p['high']
        ax.plot([l[1], h[1]] , [l[0], h[0]], c  = 'b', lw = 0.5, alpha = 0.2, zorder = 1)
    # ax = make_wcs_good(ax)
    ax.plot([] , [], c  = 'b', lw = 2, alpha = 1, zorder = 1, label = 'Radial Profile Footprint')

    ax.axis('off')


    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size = "5%", pad = 0.05)
    # fig.canvas.draw()
    # bbox = cax.get_position()
    # fig.legend(loc='upper left',
    #            bbox_to_anchor = (bbox.x0, bbox.x1),
    #            bbox_transform = fig.transFigure , 
    #            ncols = 1
    #            )

    # fig.legend( bbox_to_anchor = (0.65, 1.), frameon = False, fontsize = 12)
    return plt.gcf()





import streamlit as st


# @st.cache_resource
def plot_beam_groups(radprof): 
    plt.close()
    fig = plt.figure(figsize = (10,10))
    ax = plt.subplot(111)
    # ax = make_wcs_good(ax)
    radprof.img[radprof.img<1e10] = np.nan
    vmin = np.nanpercentile(radprof.img, 10)
    vmax = np.nanpercentile(radprof.img, 99)
    ax.imshow(radprof.img , cmap = 'Greys_r', norm = 'log', vmin = vmin, vmax = vmax, origin='lower')
    # x , y , filnum = aquilaprof.cen_x , aquilaprof.cen_y , aquilaprof.filnum
    # ax.scatter(y,x, s  = 1, c = filnum, zorder = 3, cmap = 'Spectral')
    ax.contour(radprof.skel, colors = 'k', linewidths = 0.2, alpha = 1, zorder  =1)

    xb , yb = radprof.beam_dict['cen'].T

    filnum = radprof.beam_dict['fil_index']

    ax.scatter(yb,xb, s = 30, c = filnum, marker = 'o', cmap = 'Spectral', zorder = 3)

    xl, yl = radprof.beam_dict['low']
    xh, yh = radprof.beam_dict['high']
    ax.plot([yl,yh], [xl,xh] , c = 'k', lw = 0.5, alpha = 1, zorder  = 2)
    filnum = radprof.beam_dict['fil_index']
    for indx in np.unique(filnum):
        # indx = 2
        fil_label_loc = np.where(filnum==indx)[0][0]
        fil_label_x , fil_label_y = yb[fil_label_loc] , xb[fil_label_loc]
        fil_label_text = f'{int(indx)}'
        fil_label_x , fil_label_y
        ax.text(fil_label_x, fil_label_y, fil_label_text, zorder = 4 , c = 'red', fontsize = 14, fontweight="bold" )

    ax.axis('off')
    fig.patch.set_alpha(0)


    return fig


# def plot_beam_groups(radprof, *args, **kwargs ):
#     # fig,ax = plt.subplots(1,1, 
#     #                       subplot_kw={'projection':WCS(radprof.header)},
#     #                       )
#     plt.close()
#     fig = plt.figure()
#     ax = plt.subplot(111, projection = WCS(radprof.header))
#     ax, sc = radprof.plot_props(
#         show_filid = True, 
#         ax  = ax ,
#         colorby = 'filID',
#         *args, **kwargs )
#     ax.autoscale(enable=True, axis='both')
#     # ax.axis('off')
#     # fig = plt.gcf()
#     # cbar = ax.figure.colorbar(sc,
#     #                             ax=ax,
#     #                             orientation='horizontal',
#     #                             location='top',
#     #                             pad=0.01,        # distance from the axis
#     #                             shrink=0.65, 
#     #                             aspect = 40,
#     #                             )    # make it a little narrower
#     return fig
    


def plot_all_props(radprof, *args, **kwargs ):
    ax, sc = radprof.plot_props(*args, **kwargs )
    fig = plt.gcf()
    cbar = ax.figure.colorbar(sc,
                                ax=ax,
                                orientation='horizontal',
                                location='top',
                                pad=0.01,        # distance from the axis
                                shrink=0.65, 
                                aspect = 40,
                                )    # make it a little narrower
    return fig
    


# def plot_plummer_wrapper(radprof):



def plot_premade_skl(cd, skl):
    plt.close()
    w = 9
    plt.rcParams.update({
        'axes.edgecolor' : 'k' , 
        'font.size' : 10,
        'figure.dpi' : 300 , 
    })
    fig = plt.figure(figsize = (w, w*(cd.data.shape[0]/cd.data.shape[1])))
    ax = plt.subplot(projection = WCS(cd.header))
    data_ok = cd.data 
    data_ok[data_ok<1]  = np.nan
    ax.imshow(cd.data, norm='log', cmap = 'gist_heat', vmin=0.3e22 , vmax = 1e23)
    ax.contour(skl, linewidths = 0.4, colors = 'k')
    ax = make_wcs_good(ax)
    ax.set_xlabel("l (deg)")
    ax.set_ylabel("b (deg)")
    for spine in ax.spines.values():
        spine.set_visible(False)
    # ax.axis('off')
    ax.set_frame_on(False)
    return fig


import seaborn as sns


plt.rcParams.update(
    {
    'figure.dpi' : 300,
    'font.size':12,})

def plot_selected_beam_(self):
    fig, ((ax1, ax3), (ax2,ax4)) = plt.subplots(2, 2, 
                                    sharex='col', figsize=(6,4), height_ratios=[4, 2], sharey='row')
    prof = self.med_prof_left
    r = prof[0]
    rlabel = 'r (pixels)'
    ax1.errorbar(r, prof[1],yerr=prof[2],  ls=":", zorder = 1 , ecolor='bisque')
    ax1.plot(r[:len(self.model_left)], 
                self.model_left, c='red', zorder = 2, lw = 1,
                )
    ax1.invert_xaxis()
    
    prof = self.med_prof_right
    r = prof[0]
    rlabel = 'r (pixels)'
    ax3.errorbar(r, prof[1],yerr=prof[2], label = 'Data', ls=":" , zorder = 1, ecolor='bisque')
    ax3.plot(r[:len(self.model_right)], 
                self.model_right,  zorder = 2, lw=1 , c = 'red' , 
                label = f'model | $\chi^2 = {self.red_chi[0]:.2f}$')
    
    ax3.label_outer()
    # ax3.legent()
    # ax3.invert_xaxis()
    
    ax1.set_ylabel("$N(H_2)$ X ( $10^{21})$", fontsize  = 14)

    ax2 = self.med_prof_left_fitter.plot_slope(ax = ax2)
    ax4 = self.med_prof_right_fitter.plot_slope(ax = ax4)
    ax4.label_outer()

    # ax2.legend(fontsize = 11 )
    ax2.set_xlabel(rlabel)
    ax2.set_ylabel("$dN(H_2)/ dr$", fontsize = 14)
    ax2.tick_params(axis='y', labelsize='small') #reduce fontsize

    ax3.legend(fontsize = 16 )
    plt.subplots_adjust(hspace=0.15, wspace = 0.)
    for a in [ax1,ax3, ax2 , ax4]:
        a.set_frame_on(True)
        for spine in a.spines.values():
            spine.set_edgecolor('white')
    
    return fig
    # plt.tight_layout()


def plot_selected_beam(self):
    fig, (ax1 , ax3) = plt.subplots(1, 2, 
                                    sharex='col', figsize=(10,4), sharey='row')
    prof = self.med_prof_left
    r = prof[0]
    rlabel = 'r (pixels)'
    ax1.errorbar(r, prof[1],yerr=prof[2],  ls=":", zorder = 1 , ecolor='bisque')
    ax1.plot(r[:len(self.model_left)], 
                self.model_left, c='red', zorder = 2, lw = 1,
                )
    ax1.invert_xaxis()
    
    prof = self.med_prof_right
    r = prof[0]
    rlabel = 'r (pixels)'
    ax3.errorbar(r, prof[1],yerr=prof[2], label = 'Data', ls=":" , zorder = 1, ecolor='bisque')
    ax3.plot(r[:len(self.model_right)], 
                self.model_right,  zorder = 2, lw=1 , c = 'red' , 
                label = f'model | $\chi^2 = {self.red_chi[0]:.2f}$')
    
    ax3.label_outer()
    # ax3.legent()
    # ax3.invert_xaxis()
    
    ax1.set_ylabel("$N(H_2)$ X ( $10^{21})$", fontsize  = 14)
    ax1.set_xlabel('r (Pixels)')
    

    ax3.legend(fontsize = 16 , )
    plt.subplots_adjust(hspace=0.15, wspace = 0.)
    for a in [ax1,ax3]:
        a.set_frame_on(True)
        for spine in a.spines.values():
            spine.set_edgecolor('white')
    
    return fig
    # plt.tight_layout()