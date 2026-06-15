import streamlit as st
import numpy as np

from astrools.image import Image 



from astropy.io import fits
from matplotlib import pyplot as plt


from astropy.wcs import WCS
from sutra.file_io import download_fits




plt.style.use('classic')
plt.rcParams.update({
    'figure.facecolor': 'black',
    'axes.facecolor' : "black",
    "savefig.facecolor" : "black",
    'figure.dpi' : 120,
    'font.size':18,
    'font.family' : 'monospace', 
    'axes.titlelocation': 'left',
    'axes.edgecolor' : 'bisque' , 
    'axes.titleweight': 'bold',
    'axes.titley': 1.02,
    "text.color" : 'white', 
    "axes.labelcolor" : "white" , 
    "xtick.color":"white",
    "ytick.color" : "white", 
})


# Constrain max height (e.g., 400px), preserve aspect ratio
st.markdown("""
<style>
  /* Limit image height while maintaining aspect ratio */
  .stImage img {
    max-height: 600px;
    object-fit: contain;  /* ensures whole image visible */
  }
</style>
""", unsafe_allow_html=True)


def _init_session():
    defaults = {
        "cd_image" : None , 
        "result_image" : None , 
        "cropped_image" : None, 
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


 
_init_session()              




st.sidebar.container()
st.sidebar.subheader('ISM FIlament :  Molecular cloud column density map')


# -----------------------------------------
# SIDEBAR UPLOADER 
#-----------------------------------------


# ── 5.1️⃣  Sidebar – upload + model launch ─────────────────────────────────────
st.sidebar.title("📂  Fits Image")
uploaded_image = st.sidebar.file_uploader(
    "Upload a FITS Image",
    type=["fits"],
    key="uploaded_image",
)

if uploaded_image is not None:
    im_fits = fits.open(uploaded_image)[0]
    st.session_state.im_fits = im_fits
    # st.session_state.cropped_image = None
 


st.header('Astronomical Tools (Astrools)')

image_controls , image_display , result_display = st.columns([1,3,3])

with image_controls.container(border=True):
    ra = st.number_input('Enter RA (deg)')
    dec = st.number_input("Enter DEC (deg)")

    size_x = st.number_input("Enter Height (Pixels)")
    size_y = st.number_input("Enter Width (Pixels)")

    crop_btn = st.button(
        "Crop" , 
        type="primary", 
        width='stretch'
    )
    if crop_btn:
        loc  =  ra , dec 
        size = (size_x , size_y)
        crop  = Image(st.session_state.im_fits).get_crop(loc=loc, size=size)[0]
        st.session_state.cropped_image = crop

    if st.session_state.cropped_image is not None:
        fits_buf = download_fits(
                        data=np.asarray(st.session_state.cropped_image.data, dtype='float'), 
                        _header=st.session_state.cropped_image.header
                    )
        # prob_controls.info("Skeleton map Created")
        st.download_button(
            label="Download Cropped",
            data=fits_buf.getvalue(),
            file_name="cropped.fits",
            mime="application/fits",
            # use_container_width=True,
            key="download_cropped_btn",
            width = 'stretch'
        )
     


with image_display:
    
    # st.info(st.session_state.masked_cd_map)
    if st.session_state.uploaded_image is None:
        st.info("Upload column-density FITS file in the sidebar")
    else:
        im_fig  = plt.figure()
        ax = plt.subplot(111, projection = WCS(st.session_state.im_fits.header))
        Image(st.session_state.im_fits).plot_image(cmap='YlOrBr_r', ax = ax)
        st.pyplot(im_fig)

with result_display.container():
    if st.session_state.cropped_image is not None:
        crop_fig  = plt.figure()
        ax = plt.subplot(111, projection = WCS(st.session_state.cropped_image.header))
        Image(st.session_state.cropped_image).plot_image(cmap='YlOrBr_r', ax = ax)
        st.pyplot(crop_fig)
    else : st.info('Crop Image to display here')
