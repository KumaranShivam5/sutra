# from ../scripts/astroplot import plot_wcs
import matplotlib.colors as clr
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from matplotlib import pyplot as plt
import numpy as np

import numpy as np
from scipy.signal import convolve2d
from skimage.morphology import skeletonize 
from skimage.morphology import dilation , ellipse, skeletonize , remove_small_objects
from skimage.draw import line
from skimage.filters import gaussian
from scipy.interpolate import splprep, splev
from astropy.io import fits


from scipy.signal import argrelmin , argrelmax
import math
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS


def plot_wcs(pacs_b_img , fig = None , figloc=111, norm=LogNorm(vmin=10, vmax=1000) , cbar_title = None, cmap = 'magma'):
    pacs_b_img.data[pacs_b_img.data==0] = np.nan
    if fig == None:
        fig = plt.figure(figsize=(6,6))
    wcs = WCS(pacs_b_img.header)
    # print(wcs)
    ax = fig.add_subplot(figloc, projection=wcs)
    im = ax.imshow(pacs_b_img.data, cmap=cmap, norm=norm)

    ax.set_xlabel("RA(J2000)",)
    ax.set_ylabel("Dec(J2000)", )
    ax.grid(color="black", ls='dotted', lw=1)

    # divider = make_axes_locatable(ax)
    # cax  = ax.inset_axes([1,0,0.05,1],)
    # cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cax  = ax.inset_axes([0,1,1,0.05],)
    cbar = plt.colorbar(im, cax = cax, orientation = 'horizontal')
    cax.xaxis.tick_top()
    cax.xaxis.set_label_position('top')
    if(cbar_title is None):
        cbar_title = fr"{pacs_b_img.header['BUNIT']}"
    cbar.set_label(cbar_title,  loc = 'center')


    # overlay = ax.get_coords_overlay('galactic')
    # overlay.grid(color='black', ls='dotted', lw=1)
    # overlay[0].set_axislabel('Galactic Longitude', fontsize=14)
    # overlay[1].set_axislabel('Galactic Latitude', fontsize=14)
    return fig, ax

def plot_fil(pol , indx, save_loc = None , get_fig = False):
# indx = 61
    plt.rcParams.update({'font.size':12})

    fig = plt.figure(figsize = (12,3))
    # sns.set_theme('whitegrid')
    # ax1 = plt.subplot(131, projection = WCS(pol.header))
    # ax1.imshow(pol.img, cmap = 'magma')
    # ax1.scatter(pol.cen_y, pol.cen_x, marker='.', color='white', zorder=1, s = 0.1)
    fig, ax1 = plot_wcs(pol.hdu , fig=fig , figloc=131, cbar_title=r'$N_{H_2} (cm^{-2})$' , norm = LogNorm(vmin = 1e21 , vmax = 1e22) , cmap = 'magma')
    cm_pl = clr.LinearSegmentedColormap.from_list('bin',[(0,(0,0,0,0)), (1,(0,1,1,1))])
    ax1.imshow(dilation(pol.skel , ellipse(3,3)), cmap=cm_pl)
    xlim, ylim = WCS(pol.header).world_to_pixel(SkyCoord(ra= [68,63]*u.deg , dec=[26,29.5]*u.deg))
    # ax1.set_xlim(xlim[0], xlim[1])
    # ax1.set_ylim(ylim[0], ylim[1])

    selected_filament = pol.get_filament_location(indx)
    start = selected_filament[0][0] ,selected_filament[1][0]
    end = selected_filament[0][-1] ,selected_filament[1][-1]

    # ax1.scatter(selected_filament[0] , selected_filament[1] , c = 'yellow', zorder=2, s=1)

    ax2 = plt.subplot(132 , projection = WCS(pol.header))
    ax2.plot(selected_filament[1] , selected_filament[0] ,  c = 'yellow', zorder=3, )

    exprof = pol.get_filament_profile(indx)
    for i in range(0, len(exprof['pixel locations']) , 4):
        l = exprof['pixel locations'][i]
        ax2.plot(l[1], l[0], c = 'lime', alpha = 0.5 , linewidth = 1, zorder=2)

    
    xmin , xmax = np.min(selected_filament[0]).astype('int') , np.max(selected_filament[0]).astype('int')
    ymin , ymax = np.min(selected_filament[1]).astype('int') , np.max(selected_filament[1]).astype('int')

    to_plot = pol.img[ymin:ymax, xmin:xmax]
    ax2.imshow(pol.img , zorder = 1 ,  cmap = 'magma', norm='log', )
    ax2.set_xlim(ymin-pol.N_CUT_OFF_PIX, ymax+pol.N_CUT_OFF_PIX)
    ax2.set_ylim(xmin-pol.N_CUT_OFF_PIX, xmax+pol.N_CUT_OFF_PIX)

    ax3 = plt.subplot(133)
    # plt.style.use('default')
    # for di , cd in zip(exprof['distance_array'] , exprof['colden values']):
    for i in range(0, len(exprof['pixel locations']) , 4):
        di, cd = exprof['distance_array'][i] , exprof['colden values'][i]
        cd = np.asarray(cd)
        ax3.plot(di, cd / 1e21 , c = 'white', linewidth  = 0.5)
    # plt.show()


    # Create a Rectangle patch
    rect = patches.Rectangle((ymin, xmin), abs(ymax-ymin),abs(xmax-xmin), linewidth=2, edgecolor='red', facecolor='none')
    ax1.add_patch(rect)

    # ax1.set_axis_off()
    ax1.xaxis.set_label_position("bottom")
    ax3.xaxis.tick_bottom()
    # ax1.add_beam()
    ax2.set_axis_off()
    ax3.set_ylabel(r'$N_{H2}$ [$10^{21} cm^{-2}$]')
    ax3.set_xlabel('Pixel distance from spine')
    ax3.yaxis.set_label_position("right")
    ax3.yaxis.tick_right()
    # ax3.set_xlim( -pol.cut_off_points,pol.cut_off_points)
    
    for spine in ax2.spines.values():
        spine.set_edgecolor('red')
    # ax3.set_aspect(10)
    # ax1.set_title('(a)')
    # plt.tight_layout()
    if(save_loc):
        plt.savefig(save_loc , dpi = 256, bbox_inches = 'tight')
    if(get_fig):
        return fig
    else:
        plt.show()


from matplotlib import cm
def map_to_colormap(array, colormap_name='viridis'):
  """
  Maps values in a NumPy array to a Matplotlib colormap.

  Args:
    array: A NumPy array of values.
    colormap_name: The name of the Matplotlib colormap to use (e.g., 'viridis', 'plasma', 'jet', 'coolwarm').
                   Defaults to 'viridis'.

  Returns:
    A NumPy array of RGBA colors corresponding to the input values,
    or a ValueError if the colormap name is invalid.
  """
  try:
    cmap = plt.get_cmap(colormap_name)
  except ValueError:
    raise ValueError(f"Invalid colormap name: {colormap_name}")

  norm = plt.Normalize(vmin=array.min(), vmax=array.max())  # Normalize to 0-1 range
  rgba_colors = cmap(norm(array))  # Map values to colors

  return rgba_colors
# cmp = map_to_colormap(orion.props['C']['LMD'], 'Reds')

from .Filament import Filament

from .plummer import plummer_fn
def get_beam_circ(cenx, ceny , rad):
        theta = np.linspace(-np.pi, np.pi)
        x  = cenx+rad*np.cos(theta)
        y  = ceny+rad*np.sin(theta)
        return x,y
def analysis_plot(pol, findx, pl_indx, ra = None , dec  = None):
    def get_beam_circ(cenx, ceny , rad):
        theta = np.linspace(-np.pi, np.pi)
        x  = cenx+rad*np.cos(theta)
        y  = ceny+rad*np.sin(theta)
        return x,y


    # pg.get_center()
    # plt.style.use('default')
    plt.rcParams.update({'font.size':14})
    plt.rcParams.update({'font.size':12})
    tmp = pol.skel
    # dat[0].data = dat[0].data+1e23*tmp
    # fig, ax = plot_wcs(dat[0], cbar_title=r'$N_{H_2} cm^{-2}$' , norm = LogNorm(1e21 , 1.5e22))


    fig = plt.figure( figsize = (6,8) )
    gs = GridSpec(nrows=4, ncols=2,height_ratios=[2.3,0.3,1.5,0.7])
    ax = fig.add_subplot(gs[0, :], projection = WCS(pol.header))
    ax0 = fig.add_subplot(gs[2 , 0])
    ax1 = fig.add_subplot(gs[2 , 1], sharey=ax0)
    ax2 = fig.add_subplot(gs[3 , 0])
    ax3 = fig.add_subplot(gs[3 , 1],sharey=ax2)


    # cm_pl = clr.LinearSegmentedColormap.from_list('bin',[(0,(0,0,0,0)), (1,(1,1,1,1))])
    im = ax.imshow(pol.img, cmap='magma', )

    ax.coords[1].set_axislabel("DEC(J2000)")
    ax.coords[0].set_axislabel("RA(J2000)")
    wcs = WCS(pol.header)
    # ax.imshow(dilation(pol.skel, ellipse(1,1)), cmap=cm_pl)
    axlist = [ax]
    for a in axlist:
        # ax = plt.subplot(111, projection = self.wcs)
        # im1 = plt.imshow(self.hdu.data, cmap='magma', norm=scale)
        lon = a.coords[0]
        lat = a.coords[1]
        lon.set_axislabel(' ')
        lat.set_axislabel(' ')
        lon.display_minor_ticks(True)
        lat.display_minor_ticks(True)
        lon.set_major_formatter("d.dd")
        lat.set_major_formatter("d.dd")
        lon.set_ticklabel(exclude_overlapping=True)
        lon.set_axislabel("RA(J2000)")
        lat.set_axislabel("DEC(J2000)")

    cax  = ax.inset_axes([0,1.01,1,0.05],)
    cbar = plt.colorbar(im, cax = cax, orientation = 'horizontal')
    cax.xaxis.tick_top()
    cax.xaxis.set_label_position('top')

    cbar_title = r'$N(H_2)cm^{-2}$'
    cbar.set_label(cbar_title,  loc = 'center')


    orion = Filament(pol, findx, stride = 3)
    orion.get_fil_prop(stride=3)

    cenx, ceny = orion.props['C']['center'].T
    rad = pol.pc_to_pixel(np.asarray(orion.props['C']['W_bg'])/2)

    cmp = map_to_colormap(orion.props['C']['LMD'], 'Blues')
    cmp



    for cx,cy,r , cl in zip(cenx,ceny,rad, cmp):
        x,y = get_beam_circ(cy,cx,r/2)
        ax.plot(x,y, c = 'k')
        ax.fill(x,y, c = cl, alpha = 0.8)

    # xlegend_loc , ylegend_loc = WCS(pol.header).world_to_pixel(SkyCoord(ra= [85.975]*u.deg , dec=[-2.14]*u.deg))
    # xlegend , ylegend = get_beam_circ(xlegend_loc[0] , ylegend_loc[0] , pol.pc_to_pixel(0.1)/2)
    # ax.fill(xlegend,ylegend, c = 'white', alpha=1)
    # ax.text(s=r'$radius = 0.1pc$', x = xlegend_loc+15 , y = ylegend_loc, va='center', ha = 'left', c = 'white')
    ax.grid('off')

    ax.grid(False)

    cax  = ax.inset_axes([1.01,0,0.03,1],)
    import matplotlib.colors as clr 
    norm = clr.Normalize(orion.props['C']['LMD'].min(), orion.props['C']['LMD'].max())
    sm = plt.cm.ScalarMappable(norm=norm, cmap = 'Blues')
    cbarlm = plt.colorbar(sm, cax = cax, orientation = 'vertical')
    cbarlm_title = r'$\lambda (M\odot/pc)$'
    cbarlm.set_label(cbarlm_title,  loc = 'center')
    # cax.xaxis.tick_top()
    # cax.xaxis.set_label_position('top')

    # pl_indx = 18
    x,y = get_beam_circ(ceny[pl_indx],cenx[pl_indx],rad[pl_indx]/2)
    ax.plot(x,y, c = 'white', alpha=1, lw=1)
    # ax.fill(x,y, c = 'g', alpha=0.6)

    # if ra is not None:
    #     xlim, ylim = WCS(pol.header).world_to_pixel(SkyCoord(ra= ra*u.deg , dec=dec*u.deg))
    #     ax.set_xlim(xlim[0], xlim[1])
    #     ax.set_ylim(ylim[0], ylim[1])

    pg = orion.pg_arr[pl_indx]

    ax2.set_ylabel(r'$d NH_2/d r$')
    pg.plot(which='l', ax = ax0)
    pg.plot(which='r', ax = ax1)

    
    x = orion.pg_arr[pl_indx].med_prof_right.dist
    # x = orion.get_sky_dist(x)

    params = orion.pg_arr[pl_indx].med_prof_right.get_props(dist_modifier = orion.get_sky_dist)['plm']
    plm = plummer_fn(orion.get_sky_dist(x), *params[0])
    ax1.plot(x , plm)

    for ax in fig.axes[1:]:
        ax.label_outer()
        ax.tick_params(axis="both", direction='in', length=2, which='both')
        # a.legend()
    ax0.invert_xaxis()
    ax2.invert_xaxis()
    pg.med_prof_left.plot_slope(ax =ax2, field = orion)
    pg.med_prof_right.plot_slope(ax =ax3 , field = orion)
    plt.subplots_adjust(wspace=0.01, hspace=0.15)
    # plt.tight_layout()
    ax2.set_xlabel(r'$r(pc)$')
    ax3.set_xlabel(r'$r(pc)$')
    ax3.set_ylim(-1,1)
    
    # plt.savefig('../paper-plots/orion-profile-example.png', dpi = 200, bbox_inches = 'tight')

    plt.show()
# analysis_plot(23, 6, ra = [86, 85.8], dec = [-2.23, -2.12])


def plot_radial_profile(filament, pl_indx, stride = 0.5, getfig = False):
    # orion = Filament(pol, findx, stride = stride)
    # orion.get_fil_prop(stride=stride)
    orion = filament
    pg = orion.pg_arr[pl_indx]

    fig = plt.figure( figsize = (4,4) )
    gs = GridSpec(nrows=2, ncols=2,height_ratios=[2,1])
    # ax = fig.add_subplot(gs[0, :], projection = WCS(pol.header))
    ax0 = fig.add_subplot(gs[0 , 0])
    ax1 = fig.add_subplot(gs[0 , 1], sharey=ax0)
    ax2 = fig.add_subplot(gs[1 , 0], )
    ax3 = fig.add_subplot(gs[1 , 1],sharey=ax2,)

    ax2.set_ylabel(r'$d (N_{H_2})/d r$')
    pg.plot(which='l', ax = ax0, field = orion.field)
    pg.plot(which='r', ax = ax1, field = orion.field)

    
    x = orion.pg_arr[pl_indx].med_prof_right.dist
    # x = orion.get_sky_dist(x)

    params = orion.pg_arr[pl_indx].med_prof_right.get_props(dist_modifier = orion.get_sky_dist)['plm']
    plm = plummer_fn(orion.get_sky_dist(x), *params[0])
    # ax1.plot(x , plm)
    params = orion.pg_arr[pl_indx].med_prof_left.get_props(dist_modifier = orion.get_sky_dist)['plm']
    plm = plummer_fn(orion.get_sky_dist(x), *params[0])
    # ax0.plot(x , plm)

    for ax in fig.axes[0:]:
        ax.label_outer()
        ax.tick_params(axis="both", direction='in', length=2, which='both')
        # a.legend()
    pg.med_prof_left.plot_slope(ax = ax2, field = orion)
    pg.med_prof_right.plot_slope(ax = ax3 , field = orion)
    plt.subplots_adjust(wspace=0.01, hspace=0.15)
    # plt.tight_layout()
    ax2.set_xlabel(r'$r(pc)$')
    ax3.set_xlabel(r'$r(pc)$')
    ax3.set_ylim(-1,1)
    ax0.invert_xaxis()
    ax2.invert_xaxis()
    ax2.set_xscale('log')
    ax3.set_xscale('log')

    # plt.savefig('../paper-plots/orion-profile-example.png', dpi = 200, bbox_inches = 'tight')
    if getfig: return fig
    else : plt.show()
    # plt.show()