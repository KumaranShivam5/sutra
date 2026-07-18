# -------------------------------------------------
# 1️⃣  Imports & tiny helpers
# -------------------------------------------------
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from astropy.wcs import WCS
import pandas as pd
from astropy.io import fits

from sutra.tracer.predictor import filamentIdentifier as FID


from sutra.plot_utils import plot_cd , plot_mask , plot_prob , plot_premade_skl
from sutra.profilerV2.prob2skl import run_skel

from sutra.profilerV2.prob2skl import filter_background
from sutra.file_io import load_fits, download_fits
import copy

#from streamlit_extras.stylable_container import stylable_container

from sutra.plot_utils import *

from sutra.profilerV2.radprof import RadProf


def _init_session():
    defaults = {
        "cd_map" : None , 
        "masked_cd_map" : None , 
        "mask" : None , 
        "prob_map" : None , 
        "meta_info" : {'distance' : 500 , "beam" : 36.4}, 
        "skeleton_map" : None , 
        "beamprops" : None , 
        "selected_filament_index" : None , 
        "selected_beam_index" : None, 
        "bkg_mask" : None , 
        "bkg_th" : None ,
        "masked_cd_fig" : None , 
        "props_plot_fig" : None , 
        "radprof" : None,
        "props_plot_fig" : None ,
        "comp_deets" : None, 
        "beam_prop_fig" : None , 
        "file_uploader_id" : None,
        # "meta_info"
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    st.session_state.meta_info = {'distance' : 500 , "beam" : 36.4}


 
_init_session()              



plt.style.use('classic')
plt.rcParams.update({
    'figure.facecolor': 'black',
    'axes.facecolor' : "black",
    "savefig.facecolor" : "black",
    'figure.dpi' : 120,
    'font.size':18,
    'font.family' : 'monospace', 
    'axes.titlelocation': 'left',
    'axes.edgecolor' : 'k' , 
    'axes.titleweight': 'bold',
    'axes.titley': 1.02,
    "text.color" : 'white', 
    "axes.labelcolor" : "white" , 
    "xtick.color":"white",
    "ytick.color" : "white", 
})

st.markdown("""
        <style>
        /* Example: Reduce top padding of the main content block */
            *{font-family: Arial;}
        .block-container {
            padding-top: 2rem; /* Adjust as needed */
        }
            
        .stMarkdown{
            margin-top:0rem !important;
            }
        .stMainBlockContainer{
            padding-left: 10rem !important;
            padding-right : 10rem !important;
            padding-top:1rem !important;
            padding-bottom:0rem !important;
            margin-bottom:0rem !important;
            }
        body {
            margin-bottom:0rem !important;
            padding-bottom:0rem !important;
            }
        .stAppHeader{ 
        
            }
        .stVerticalBlock{
            gap:2rem !important;
        }
        .stSidebarHeader{
            margin-bottom : 0 !important; 
            height:0 !important;
        }
        /* Example: Reduce gap between elements in a specific column */
        [data-testid="column"]:nth-of-type(1) [data-testid="stVerticalBlock"] {
            gap: 0rem; /* Remove gap */
        }
        code {color:#fbd48a !important;}
        
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMardownContainer"] p {
            font-weight : 500 ; 
            font-size : 1.2rem;
            }    
        
            
        [data-testid="stAlertContentInfo"] p{
                font-size: 0.9rem !important;  /* adjust as needed */
                line-height: 1.2;
            }
        [data-testid="stAlertContentInfo"] .katex{
            font-size : 0.9rem;
            }
        
            img {
            max-height:100 !important;
            object-fit : contain;
            }

            @media print {
                @page {
                    margin-bottom : 5mm;
                    margin-top : 5mm;
                }
            .block-container {
                padding-top : 0rem , 
            padding-bottom : 0rem;
            }
            }
            
        </style>
    """, unsafe_allow_html=True)
st.markdown("""
<style>
  /* Limit image height while maintaining aspect ratio */
  .stImage img {
    max-height: 500px;
    object-fit: contain;  /* ensures whole image visible */
  }
</style>
""", unsafe_allow_html=True)



breakpoint = "1920px"
st.markdown(f"""

    <style>
            @media (max-width : {breakpoint}) {{
            [data-testid="stVerticalBlock"]{{
                flex-direction:column !important;
                min-width:100px;
            }}
            }}
            [data-testid="stVerticalBlock"]:has([data-testid="stMarkdownContainer"] .wrap-col-marker)[data-testid="column"]{{
            width:100% !important;
            flex: 1 1 100% !important;
            }}
            </style>
""", unsafe_allow_html=True)


st.set_page_config(
    
    page_title="SUTRA - Filament tools",
    layout="wide",
    initial_sidebar_state="expanded",
)

from pathlib import Path

BASE_DIR = Path(__file__).parent  # directory where app.py lives



st.subheader(':primary[:material/draw: Filament Identification  |  Tracing filament Skeleton using U-Net Model]')
identification_window = st.container(border  = False , width = 'stretch')

HEADING_BORDER=True


with st.container():
    # st.markdown('<div class="wrap-col-marker">wrapping col</div>', unsafe_allow_html=True)
    cd_window , prob_window  = st.columns(
    [1,1], border =True , 
    width='stretch')
    # tangent_plot , group_plot = st.columns(
    # [1,1], border =True , 
    # width='stretch')


with cd_window.container(border=HEADING_BORDER):
    st.markdown("### `1. Upload Image | Preprocessing`")
    st.markdown(" <span style='color:#00e5e5;'>Upload FITS imge(CD, Flux density) :material/arrow_forward_ios: Set observation parameters :material/arrow_forward_ios: background masking [3] :material/arrow_forward_ios: border-masking :material/arrow_forward_ios: global normalisation</span>", unsafe_allow_html = True, text_alignment = 'justify')

with prob_window.container(border=HEADING_BORDER):
    st.markdown("### `2. Trace Filament Crest`")
    st.markdown("<span style='color:#00e5e5;'>Masked map split into overlapping chunks :material/arrow_forward: Local normalisation :material/arrow_forward: U-Net on chunks :material/arrow_forward: crest probability map :material/arrow_forward: reassamble map :material/arrow_forward: Morphological skeletonisation</span>", unsafe_allow_html = True, text_alignment = 'justify')

# with tangent_plot.container(border=HEADING_BORDER):
#     st.markdown("### `3. Compute Radial Profile`")
#     st.markdown("<span style='color:#00e5e5;'>Bezier curves skeleton smootheining :material/arrow_forward_ios: Local normnals computation :material/arrow_forward_ios: Radial Profiles extraction :material/arrow_forward_ios: Shift center of profiles to maxima</span>", unsafe_allow_html = True, text_alignment = 'justify')



cd_col_controls , cd_col_display = cd_window.columns([2,3], border=False)
prob_controls , prob_plots = prob_window.columns([2,3],
                                                 border=False)
# rad_controls , rad_prof_window = tangent_plot.columns([2,3], border=False)
#---------------------
# st.subheader(':primary[:material/design_services: Filamrnt Characterisation | Measure Filament properties]')

characterisation = st.container(border = False , width='stretch')


#---------






# st.sidebar.info(st.session_state.bkg_th)

cd_window_status = cd_window.empty()


def reset_session(del_keys):
    # for k in del_keys:
    #     st.session_state[k] = None
    # # cd_window_status.empty()
    # print('RESET')
    pass
    # cd_window_status.error('resetting everything')
    



    # st.rerun()
uploaded_cd = st.sidebar.file_uploader(
    "Upload a FITS File",
    type=["fits", "gz"],
    key="upload_cd", 
    accept_multiple_files=False,
    max_upload_size = 250,
    # on_change = reset_session(["mask", "skeleton_map","beamprops","bkg_mask","radprof" ])
    )


if uploaded_cd is not None:
    file_id_now = f"{uploaded_cd.name}_{uploaded_cd.size}"
    if file_id_now == st.session_state['file_uploader_id']:
        reset_session(["mask", "prob_map", "skeleton_map","beamprops","bkg_mask","radprof" ])
        print('reset yes')
    st.session_state['file_uploader_id'] = file_id_now

with cd_window:
    # cd_window_status.success('not resetting')


    if uploaded_cd is not None:
        cd_fits = fits.open(uploaded_cd)[0]
        cd_fits.data[cd_fits.data<0.001] = np.nan
        cd_fits.data[cd_fits.data<0.001] = np.nan
        cd_fits.data[cd_fits.data<1e18] = np.nan 
        cd_fits.data[cd_fits.data>1e28] = np.nan 
        st.session_state.cd_map = cd_fits
    else:
        with open(BASE_DIR / '../data' / 'vela-cropped-2.fits','rb') as f:
            cd_fits = fits.open(f)[0]
            cd_fits.data[cd_fits.data<0.001] = np.nan
            cd_fits.data[cd_fits.data<0.001] = np.nan
            cd_fits.data[cd_fits.data<1e18] = np.nan 
            cd_fits.data[cd_fits.data>1e28] = np.nan 
            st.session_state.cd_map = cd_fits
    
        
    # st.info(st.session_state.masked_cd_map)
    if st.session_state.cd_map is None:
        st.info("Upload column-density FITS file in the sidebar")
    else:

        cd = st.session_state.cd_map
        

        with cd_col_controls:
            # with st.contianer():
            # st.subheader("CD Properties")
            with st.expander('Observation Parameters', expanded=True):
                distance = st.number_input(
                "Distance (pc)", min_value=1.0, value=700.0, step=1.0, key="meta_dist"
                )
                beam = st.number_input(
                    "Beam (arcsec)", min_value=1.0, value=36.4, step=0.1, key="meta_beam"
                )
                # Store the dict *once* – it will be re‑used later without re‑creating it
                st.session_state.meta_info = {"distance": st.session_state.meta_dist
                                                , "beam": st.session_state.meta_beam}


        # with st.expander("Background Mask", expanded=True):

            _ , auto_bkg_th = filter_background(cd.data)

            # if st.session_state.bkg_th is None:
            #     _ , auto_bkg_th = filter_background(cd.data)
            #     st.info(f"Background Threshold: {auto_bkg_th:1.2e} $N(H_2)/cm^{2}$")
            #     bkg_mask , bkg_th = filter_background(
            #         cd.data, val= float(np.log10(auto_bkg_th)))
            #     st.session_state.bkg_th  = auto_bkg_th
            #     st.session_state.bkg_mask = bkg_mask
            st.sidebar.info(np.nanmax(np.log10(cd.data)))
            with st.form(key="background_compute"):
                bkg_th_user = st.slider(
                    label="Select Custom value for background",
                    min_value=float(1.1*np.nanmin(np.log10(cd.data))),
                    max_value=float(np.nanmax(np.log10(cd.data))) , 
                    # step=0.1,
                    value=float(np.log10(auto_bkg_th)),
                    key="bkg_th_user"
                    )
             
                compute_cd_mask_btn = st.form_submit_button("Compute mask" ,width='stretch')
                if compute_cd_mask_btn:
                    bkg_mask , bkg_th = filter_background(
                    cd.data, val= float(st.session_state.bkg_th_user))
                    st.session_state.bkg_th  = bkg_th
                    st.session_state.bkg_mask = bkg_mask

                    st.sidebar.info(st.session_state.bkg_th_user)
                    st.sidebar.info(bkg_mask.sum())

            

            if st.session_state.bkg_mask is not None:
                # st.info(st.session_state.meta_info)
                masked_cd = fits.PrimaryHDU(data = cd.data , header = cd.header)
                masked_cd.data*=st.session_state.bkg_mask
                st.session_state.masked_cd_map = masked_cd

        if st.session_state.masked_cd_map is not None:
            if st.session_state.skeleton_map is not None:
                fig = plot_cd(st.session_state.masked_cd_map, skeleton=st.session_state.skeleton_map)
            else:
                fig = plot_cd(st.session_state.masked_cd_map)
        else:
            fig = plot_cd(st.session_state.cd_map)

        cd_col_plot_placeholder = cd_col_display.empty()
        cd_col_plot_placeholder.pyplot(fig, use_container_width=True)
        # st.caption(''' Fig.2 : CD map from Vela-C molecular cloud [4].''')


import streamlit.components.v1 as components

def simulate_esc_close():
    components.html(
        """
            <script>
            parent.document.querySelector('[aria-label="Close"]').click();
            </script>
        """,
        height =  0 , 
        width = 0 ,
    )


@st.dialog('Estimated Compute Time ')
def _estimate_comp_tim_fid(fid, cd_map , bs):
    # def _int_fn():
    # @st.fragment
    # def _fragment_function():
    # if st.session_state.comp_deets is None: # makes it run once
    with st.spinner("Estimating Computation Time"):
        comp_deets = fid.computation_cost(cd_map , batch_size = bs, window_overlap_frac = st.session_state.overlap_frac)
    st.session_state.comp_deets = comp_deets
    st.info(f'''
        * Input image to be divided into : **{st.session_state.comp_deets['Total patches']}** chunks
        * Approximate Time req. : {st.session_state.comp_deets['Estimated total (s)']} s''',)
    b1,b2 = st.columns(2)
    cancel_btn = b1.button('Cancel', type='secondary', width='stretch',)
    run_btn = b2.button('Continue', type='primary', width='stretch')
    if cancel_btn:simulate_esc_close()
    if run_btn :#or (st.session_state.prob_map is None): # run once if page is refreshed or when user clicks
        with st.spinner(f"Applying U-Net model on {st.session_state.comp_deets['Total patches']} chunks"):
            prob_map = fid.predict(st.session_state.masked_cd_map, 
                                window_overlap_frac = st.session_state.overlap_frac, 
                                batch_size = 1024, 
                                n_jobs = 1)
        st.session_state.prob_map = prob_map
            # st.rerun(scope='fragment')
        simulate_esc_close()
        st.rerun()



with prob_window:

    if st.session_state.masked_cd_map is None:
        st.info("Upload CD to Continue")
    else:
        # with prob_controls.container(width='stretch', border=False):
        with prob_controls.form(key = 'run_model_form' ,  width='stretch', border=True):
            # overlap_frac = 0.8
            model_name = st.selectbox(
                "Choose Tracer model",
                options=["HGBS", "HiGAL"],   # keep the names you actually support
                key="model_selector",
            )
            fid = FID(model_name)
            # with st.expander('Model Details', expanded=False):
            #     st.write(fid.get_model_details())
            overlap_frac = st.slider(
                label="Chunk-Overlap(%) :" , 
                min_value=0.60, 
                max_value=0.95,
                step = 0.01, 
                value=0.85, 
                key="overlap_frac",
                help=f"Input image will be divided int chunks of size:{fid.model.input_shape[0]}. Adjust the fraction of overlap between adjecent chunks. Higher fraction > more chunks > more computaiton time. **85% is optimal.**"
                )
            
                # st.write(comp_deets)
            run_btn = st.form_submit_button(":material/batch_prediction: Run model", key="run_model_btn", width =  'stretch')
            if run_btn: 
                _estimate_comp_tim_fid(fid,
                                        cd_map=st.session_state.masked_cd_map, 
                                        bs= 1024 )
                # st.rerun()

        if st.session_state.prob_map is None:
            prob_plots.info("Select and Run model to generate Crest probability map")
        else:
            prob_fig = plot_prob(
                st.session_state.prob_map , 
                header=st.session_state.masked_cd_map.header 
            )

            prob_plots_prob_map_ph = prob_plots.empty()

            prob_plots_prob_map_ph.pyplot(prob_fig)
            # skeleton_btn = prob_controls.button("Skeletonize", width='stretch')
            # if skeleton_btn:
            skel = run_skel(
                    st.session_state.prob_map, 
                    th_max = 0.3, 
                    th_min = 0.17, 
                    bkg_mask=st.session_state.bkg_mask, 
                    beam_size = 3, 
                    prune = True, 
                    convolve_map = False,
                    )
            st.session_state.skeleton_map = skel
        # st.rerun()



        if st.session_state.skeleton_map is not None:
            fits_buf = download_fits(
                    data=np.asarray(st.session_state.skeleton_map, dtype="int"),
                        _header=st.session_state.masked_cd_map.header
                )
            prob_controls.download_button(
                label="Download skeleton",
                data=fits_buf.getvalue(),
                file_name="skeleton.fits",
                mime="application/fits",
                # use_container_width=True,
                key="download_skel_btn",
                width = 'stretch', 
                type  = 'primary',
                disabled = st.session_state.skeleton_map is None
            )
            cd_fig_with_skl = plot_cd(
                st.session_state.masked_cd_map, 
                skeleton=st.session_state.skeleton_map)
            cd_col_plot_placeholder.empty()
            cd_col_plot_placeholder.pyplot(cd_fig_with_skl)
                
        # else : st.write("Skeleton Not available") 
        # st.caption('''Fig.3 : Output from U-Net model trained on DisPerSE+getsf skeleton ''')

                # characterisation.page_link("pages/2_Properties.py", label="Go to Radial Profiling",)



