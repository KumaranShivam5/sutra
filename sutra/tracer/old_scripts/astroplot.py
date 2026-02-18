from matplotlib import pyplot as plt 
from astropy.wcs import WCS 
from matplotlib.colors import LogNorm
from astropy import units as u
import numpy as np



from astropy import constants as const


from mpl_toolkits.axes_grid1 import make_axes_locatable
def plot_prediction(cd_img, true_skl = None, pred_skl = None, exp_name=None, nrows=2,ncols=6, process_skl = False):

#     cmaps = {
#         'cloud' : 'inferno',
#         'true_skl' : 'binary_r',
#         'pred_skl' : 'Wistia_r',
#         'hist': 'black'
#     }
    cmaps = {
        'cloud' : 'inferno',
        'true_skl' : 'binary_r',
        'pred_skl' : 'Greys',
        'hist': 'black'
    }

    from skimage.morphology import dilation
    from matplotlib.colors import Normalize
    indx_arr = np.arange(nrows*ncols)
    np.random.shuffle(indx_arr)
    cd_img = np.squeeze(cd_img)
    true_skl = np.squeeze(true_skl)
    pred_skl = np.squeeze(pred_skl)
    cd_img = [cd_img[i] for i in indx_arr]
    pred_skl = [pred_skl[i] for i in indx_arr]
    true_skl = [true_skl[i] for i in indx_arr]
    import pandas as pd 
    import os 
    from scipy.ndimage import histogram
    # nrows , ncols = 2,6
    img_list = cd_img
    fig, ax = plt.subplots(nrows=nrows,ncols=ncols, figsize=(ncols*3 , nrows*3), sharex=True, sharey=True, )
    ax = np.ravel(ax)
    # plt.imshow()
    # for i, ax in zip()
    for i in range(len(ax)):
        a = ax[i]
        if(i%2):
            true_im = pred_skl[i-1]
            im = img_list[i-1]
            skl_clr = cmaps['pred_skl']
            alpha_pred =  Normalize(0.01, 0.1,clip=True)(np.abs(true_im))
            # true_im = alpha_pred
        else:
            im = img_list[i]
            true_im = true_skl[i]
            skl_clr = cmaps['true_skl']
            alpha_pred = true_im
        # a , im , pred_im, true_im = ax[i],img_list[i], pred_skl[i] , true_skl[i]
        imx = a.imshow(im, origin='lower', cmap = cmaps['cloud'])
        make_axes_locatable(a)
        cax  = a.inset_axes([0.0,-0.05,1,0.05],)
        fig.colorbar(imx, cax=cax, orientation='horizontal')

        a2 = a.twinx().twiny()
        im_min , im_max = np.min(im) , np.max(im)
        hist_range = np.linspace(im_min, im_max,  100 )
        hist_val = histogram(im, im_min , im_max, bins=100)
        a2.plot(hist_range, hist_val, color=cmaps['hist'], linewidth=0.5)

        # impred = a.imshow(y_pred, alpha = y_pred ,)
        # alpha_pred = Normalize(0.1, 0.2,clip=True)(np.abs(pred_im))
        # alpha_pred = np.clip(alpha_pred , 0.4,1)
        if(process_skl):
            alpha_pred = dilation(alpha_pred,)
            # pred_im = dilation(pred_im)
            true_im = dilation(true_im)
        # impred = a.imshow(alpha_pred, alpha = alpha_pred, cmap='binary', origin='lower')
            imtrue = a.imshow(alpha_pred, alpha = alpha_pred.astype('float'), cmap = skl_clr, origin='lower')
        else:
            imtrue = a.imshow(true_im, alpha = true_im.astype('float'), cmap = skl_clr, origin='lower')

        if(i%2):
            cax  = a.inset_axes([1.0,0.0,0.05,1],)
            fig.colorbar(imtrue, cax=cax, orientation='vertical')

        # a.set_yticks([])
        a.set_xticks([])
        a2.set_xticks([])
        a2.set_yticks([])
        a.label_outer()

    plt.suptitle(f'{exp_name}', ha='left')
    plt.tight_layout()
    return fig


from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.gridspec as gs

def plot_preprocessing(img_loc, exp_name=None, nrows=2,ncols=6):
    import pandas as pd 
    import os 
    from scipy.ndimage import histogram
    # nrows , ncols = 2,6
    indx_arr = np.arange(nrows*ncols, dtype=int)
    np.random.shuffle(indx_arr)
    # print(indx_arr)
    # indx_arr = np.array(indx_arr, dtype=int)

    img_list = [np.load(f'{img_loc}/CD/{f}') for f in pd.Series(os.listdir(f'{img_loc}/CD'))]
    img_list = np.array([img_list[i] for i in indx_arr])

    tg_list = [np.load(f'{img_loc}/skeleton/{f}') for f in pd.Series(os.listdir(f'{img_loc}/skeleton'))]
    tg_list = np.array([tg_list[i] for i in indx_arr])
    fig, ax = plt.subplots(nrows=nrows,ncols=ncols, figsize=(ncols*3 , nrows*3), sharex=True, sharey=True, layout='constrained')
    ax = np.ravel(ax)
    for a , im , tg in zip(ax,img_list, tg_list):
        imx = a.imshow(im, cmap='inferno')
        make_axes_locatable(a)
        cax  = a.inset_axes([0.0,-0.05,1,0.05],)
        fig.colorbar(imx, cax=cax, orientation='horizontal')

        a2 = a.twinx().twiny()
        im_min , im_max = np.min(im) , np.max(im)
        hist_range = np.linspace(im_min, im_max,  100 )
        hist_val = histogram(im, im_min , im_max, bins=100)
        a2.plot(hist_range, hist_val, color='white')

        # a.set_yticks([])
        a.imshow(tg, alpha=tg.astype('float'))
        a.set_xticks([])
        a2.set_xticks([])
        a2.set_yticks([])
        a.label_outer()
    # ax[0].set_title('exp')
    plt.suptitle(f'{exp_name}', ha='left')
    return fig




def quick_show(img, header=None, norm=LogNorm(vmin=0.01, vmax=10)):
    if(header):
        ax1 = plt.subplot(111, projection=WCS(header))
    else:
        ax1 = plt.subplot(111, projection=WCS(img.header))
    if(type(img)==np.ndarray):
        img_data = img 
    else:
        img_data = img.data
    img_data[img_data==0] = np.nan
    ax1.imshow(img_data, origin='lower', norm=norm, cmap='copper')


def plot_wcs(pacs_b_img , fig = None , figloc=111, norm=LogNorm(vmin=10, vmax=1000) ):
    pacs_b_img.data[pacs_b_img.data==0] = np.nan
    if fig == None:
        fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(figloc, projection=WCS(pacs_b_img.header))
    im = ax.imshow(pacs_b_img.data, cmap='inferno', norm=norm)

    ax.set_xlabel("Right Ascension", fontsize=12)
    ax.set_ylabel("Declination", fontsize=12)
    # ax.grid(color="white", ls='dotted', lw=2)

    # cbar = plt.colorbar(im, pad=0.1)
    # cbar.set_label(f"{pacs_b_img.header['BUNIT']}", size=16)

    # overlay = ax.get_coords_overlay('galactic')
    # overlay.grid(color='black', ls='dotted', lw=1)
    # overlay[0].set_axislabel('Galactic Longitude', fontsize=14)
    # overlay[1].set_axislabel('Galactic Latitude', fontsize=14)
    return fig, ax



def make_wcs_good(ax):
    axlist = [ax]
    for a in axlist:
        lon = a.coords[0]
        lat = a.coords[1]
        lon.set_axislabel(' ')
        lat.set_axislabel(' ')
        lon.display_minor_ticks(True)
        lat.display_minor_ticks(True)
        lon.set_major_formatter("d.dd")
        lat.set_major_formatter("d.dd")
        lon.set_ticklabel(exclude_overlapping=True)
    ax.set_xlabel("RA (J2000)",)
    ax.set_ylabel("Dec (J2000)", )
    return ax

# import matplotlib.colors as mcolors
import matplotlib.colors as mcolors
def bin_cmap(clr):
  """Creates a binary color palette for matplotlib where 0s are transparent 
  and 1s are red.

  Returns:
    A matplotlib ListedColormap object representing the binary palette.
  """

#   colors = [(0, 0, 0, 0), clr]  # Red is (1, 0, 0)
  cmap_name = 'binary_red'
  cmap = mcolors.LinearSegmentedColormap.from_list('bin',[(0,(0,0,0,0)), (1, clr)])
  return cmap


def plot_skl(pacs_b_img , fig = None , figloc=111, norm=LogNorm(vmin=10, vmax=1000) , cbar_title = None, cmap = 'magma'):
    if fig == None:
        fig = plt.figure(figsize=(6,6))
    wcs = WCS(pacs_b_img.header)
    # print(wcs)
    ax = fig.add_subplot(figloc, projection=wcs)
    im = ax.imshow(pacs_b_img.data,cmap=cmap)

    ax.set_xlabel("RA(J2000)",)
    ax.set_ylabel("Dec(J2000)", )
    ax.grid(color="black", ls='dotted', lw=1)

    return fig, ax



from astropy.io import fits
from astropy import units as u
class field():
    def __init__(self,fname, hdu=1, name=None) -> None:
        self.hdu = fits.open(fname)[hdu]
    def resolution(self):
        imwcs = WCS(self.hdu.header).proj_plane_pixel_scales()
        return [imwcs[0].to(u.arcsec), imwcs[1].to(u.arcsec)]