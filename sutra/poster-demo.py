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
            padding-left: 4rem !important;
            padding-right : 4rem !important;
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
            gap:1rem !important;
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


st.set_page_config(
    
    page_title="SUTRA - Filament tools",
    layout="wide",
    initial_sidebar_state="expanded",
)

from pathlib import Path

BASE_DIR = Path(__file__).parent  # directory where app.py lives



st.sidebar.container()
st.sidebar.subheader('ISM FIlament :  Molecular cloud column density map')


# -----------------------------------------
# SIDEBAR UPLOADER 
#-----------------------------------------


# ── 5.1️⃣  Sidebar – upload + model launch ─────────────────────────────────────
st.sidebar.title("📂  Column‑density map")


title_box , logo_box = st.columns([4,1], width='stretch', vertical_alignment='center')

title_box.markdown(f"<h2> <span style = 'color:white;font-weight:normal !important;font-size:3rem;'>Sutra : </span> <span style = 'color:#fbd48a;font-weight:normal;'> An ML based framework for Interstellar medium filament identification and beam-level characterisation</span></h2> ", unsafe_allow_html=True)

l1,l2 , l3  = logo_box.columns([1,1,1], width='stretch', vertical_alignment='center')
l1.image(str(BASE_DIR/'poster_images'/'SSD.png'), width=120)
l2.image(BASE_DIR/'poster_images'/'SAC.png', width=120)
l3.image(BASE_DIR/'poster_images'/'ISRO.png', width=120)


 

top_info , use_case = st.columns([7,2], border = True)


with top_info.container(width='stretch', border=False):
    qr , auth = st.columns([2,20], border=True, vertical_alignment='center')
    qr.image(BASE_DIR/'poster_images'/'QR.png', width='stretch')
    auth.markdown(r'''<span style='font-size:1.5rem;color:#fbd48a;'>Shivam Kumaran, Ushasi Bhowmick, Vipin Kumar, Manish Chauhan, Munn V. Shukla, Mehul R Pandya</span><br> <span style='font-size:1.1rem;'>Space Sciences Division, Space Applications Centre, ISRO, Ahmedabad, India</span>''', unsafe_allow_html=True)



# intro_box = st.container()
# abstract_box , examples_box = st.columns([5,3], border=True, )

with top_info.container(width='stretch', border=False):
    # st.subheader("Introduction")
    st.markdown("<span style='font-size:1.5rem;color:#fb8a8aff;font-weight:bold;'>Introduction</span> : Filamentary structures in molecular clouds play a central role in star formation, yet large-scale identification and characterisation remains difficult as existing tools require manual parameter tuning and perform inconsistently across varying backgrounds. Sutra employs a U-Net trained on consensus skeletons from DisPerSE[1] and GETSF[2], enabling parameter-free detection across a wide range of column densities. Each candidate skeleton segment is evaluated through Plummer profile fitting[3] at the beam scale; segments inconsistent with cylindrical filament profiles are rejected, ensuring the final skeleton represents physically meaningful structures. Beam-level radial profiles then yield local measurements of linear mass density, width, contrast and p-index, capturing property variations along filament axes that filament-averaged profiles would obscure." , 
                text_alignment='justify', unsafe_allow_html=True
                )
USE_BOX_STYLE = """
{
    background: #042124;
    border-color: black;
}
"""
use_case_container = top_info.container(border=False)
with use_case_container:
    st.markdown(
        """
        <style>
        .sutra-usecases {
            display: flex;
            flex-direction: row;
            gap: 0.5rem;
            width: 100%;
            margin-bottom: 0.5rem;
            margin-top: 0.5rem;
        }
        .sutra-usecases .box {
            background: #042124;
            border: 1px solid black;
            border-radius: 0.5rem;
            padding: 1rem;
            flex: 1;
        }
        .sutra-usecases .box.header {
            # flex: 0.5;
            font-size : 3rem;
        }
        .sutra-usecases .box h5 {
            color: bisque;
            margin: 0 0 0.25rem 0;
            font-size: 1rem;
        }
        .sutra-usecases .box p {
            margin: 0;
            font-size: 1rem;
            color: inherit;
        }
        .sutra-usecases .box:nth-child(1) { flex: 5; }
        .sutra-usecases .box:nth-child(2) { flex: 8; }
        .sutra-usecases .box:nth-child(3) { flex: 7; }
        .sutra-usecases .box:nth-child(4) { flex: 5; }
        .sutra-usecases .box:nth-child(5) { flex: 6; }
        </style>

        <div class="sutra-usecases">
            <div class="box header">
                <h5>Sutra key use cases</h5>
            </div>
            <div class="box">
                <p><strong>Hub-Filament Systems</strong>: Filament properties variations reveal mass flow in cluster-forming hubs.</p>
            </div>
            <div class="box">
                <p><strong>Filament Fragmentation</strong>: Beam-level profiles trace core spacing and instability.</p>
            </div>
            <div class="box">
                <p><strong>Accretion</strong>: Widths and gradients link filaments to protostellar growth.</p>
            </div>
            <div class="box">
                <p><strong>Large-Scale Analysis</strong>: Parameter-free skeletons across varied backgrounds.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # u5.markdown("**Comparative Studies**: Consistant properties computation across clouds for ISM studies.", text_alignment='justify')

# use_case_image , use_case_text = use_case.columns([2.2,2], border=False, )


use_case_image  = use_case.container(border=False, )

use_case_image.image(BASE_DIR/'poster_images'/'monr2-filament-example-2.png', width=430)
use_case_image.caption(body=''' **Fig 1: Sutra application on the Mon-R2 hub-filament system [4]**. Scatter color and size indicate filament density and width variations.''', text_alignment='justify')
# use_case_text.markdown('''### Use Cases of Sutra
# * **Hub-Filament Systems** : spatially resolved mass-flow towards hub and stability
# * **Fragmentaiton Analysis** : filament local structural variations inform core spacing and fragmenation physics
# * **Accretion and Core Growth** : Resolved widths along filament helps link structural evolution to protostellar mass accretion
# * **Large-Scale, Parameter-Free Identification** : Sutra's automaic skeleton extraction works across varied backgrounds without manual threshold tuning''')


# use_case_text.markdown('''##### Sutra Key Use cases:
# * **Hub-Syatems** : Filament properties variations revel mass flow in cluster-forming hubs.
# * **Fragmentation** : Beam-level profiles trace core spacing and instability
# * **Accretion** : Widths and gradients link filaments to protostellar growth.
# * **Large-Scale Analysis**: Parameter-free skeletons across varied backgrounds.
# * **Environmental Comparisons**: Consistant properties computation across clouds for ISM studies.
# ''', text_alignment='justify')

# -------------------------------------------------
# Main layout – three columns that stay static
# -------------------------------------------------

st.subheader(':primary[:material/draw: Filament Identification  |  Tracing filament Skeleton using U-Net Model]')
identification_window = st.container(border  = False , width = 'stretch')

HEADING_BORDER=True

cd_window , prob_window , tangent_plot = identification_window.columns(
    [1.3,1.2,1], border =True , 
    width='stretch')
with cd_window.container(border=HEADING_BORDER):
    st.markdown("### `1. Upload Image | Preprocessing`")
    st.markdown(" <span style='color:#00e5e5;'>Upload FITS imge(CD, Flux density) :material/arrow_forward_ios: Set observation parameters :material/arrow_forward_ios: background masking [3] :material/arrow_forward_ios: border-masking :material/arrow_forward_ios: global normalisation</span>", unsafe_allow_html = True, text_alignment = 'justify')

with prob_window.container(border=HEADING_BORDER):
    st.markdown("### `2. Trace Filament Crest`")
    st.markdown("<span style='color:#00e5e5;'>Masked map split into overlapping chunks :material/arrow_forward: Local normalisation :material/arrow_forward: U-Net on chunks :material/arrow_forward: crest probability map :material/arrow_forward: reassamble map :material/arrow_forward: Morphological skeletonisation</span>", unsafe_allow_html = True, text_alignment = 'justify')

with tangent_plot.container(border=HEADING_BORDER):
    st.markdown("### `3. Compute Radial Profile`")
    st.markdown("<span style='color:#00e5e5;'>Bezier curves skeleton smootheining :material/arrow_forward_ios: Local normnals computation :material/arrow_forward_ios: Radial Profiles extraction :material/arrow_forward_ios: Shift center of profiles to maxima</span>", unsafe_allow_html = True, text_alignment = 'justify')



cd_col_controls , cd_col_display = cd_window.columns([2.5,3], border=False)
prob_controls , prob_plots = prob_window.columns([2.5,3],
                                                 border=False)
rad_controls , rad_prof_window = tangent_plot.columns([2.5,3], border=False)
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
    max_upload_size = 50,
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
        with open(BASE_DIR / 'data' / 'vela-cropped-2.fits','rb') as f:
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
            with st.expander('Observation Parameters', expanded=False):
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

            st.info(f"Background Threshold: {auto_bkg_th:1.2e} $N(H_2)/cm^{2}$")

            bkg_mask , bkg_th = filter_background(
                cd.data, val= float(np.log10(auto_bkg_th)))
            st.session_state.bkg_mask = bkg_mask
        
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

        # cd_col_plot_placeholder = cd_col_display.empty()
        cd_col_display.pyplot(fig, use_container_width=True)
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
                )
            
                # st.write(comp_deets)
            run_btn = st.form_submit_button(":material/batch_prediction: Run model", key="run_model_btn", width =  'stretch')
            if run_btn: 
                _estimate_comp_tim_fid(fid,
                                        cd_map=st.session_state.masked_cd_map, 
                                        bs= 1024 )

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
            # cd_fig_with_skl = plot_cd(
            #     masked_cd, 
            #     skeleton=st.session_state.skeleton_map)
            # cd_col_plot_placeholder.empty()
            # cd_col_plot_placeholder.pyplot(cd_fig_with_skl)
                
        # else : st.write("Skeleton Not available") 
        # st.caption('''Fig.3 : Output from U-Net model trained on DisPerSE+getsf skeleton ''')

                # characterisation.page_link("pages/2_Properties.py", label="Go to Radial Profiling",)





characterisation.subheader(':primary[:material/design_services: Filament Characterisation  |  Beam-Level Measurements of Filament properties]')
props_plot , filament_beam_plot = characterisation.columns([2,4], border=True)
if st.session_state.radprof is None:
    characterisation.info('Create Radial Profile in `compute Radial Profile` to continue Characterisation')

# with props_plot.container(border=HEADING_BORDER):


def _execute_rad_prof_creation():
    print('sdsdsdsdsdsdsdsds')
    st.session_state.radprof = RadProf(
            img=st.session_state.masked_cd_map , 
            skeleton= st.session_state.skeleton_map , 
            meta_info = st.session_state.meta_info, 
        )
    st.session_state.radprof.tangents(
        ks = st.session_state.stride+2, 
        stride = st.session_state.stride)
    with st.spinner("Reordering"):
        st.session_state.radprof.reorder()
    with st.spinner("Smoothing Spline"):
        st.session_state.radprof.spline_smooth()
    with st.spinner("Set Cut-off points"):
        st.session_state.radprof.cut_off_points(
            st.session_state.rad_cut_dist)
    with st.spinner("Creating radial profiles"):
        st.session_state.radprof.create_rad_profile_single_thread()




with tangent_plot:
    # st.spinner
    if st.session_state.skeleton_map is not None:
        
        with rad_controls.form("rad_profiling_ctrls"):
            rad_cut_dist = st.slider(
                "Radial cutoff (Pc)",
                min_value = st.session_state.meta_info['distance']/1000 , 
                max_value = 5*st.session_state.meta_info['distance']/1000 ,
                value = st.session_state.meta_info['distance']/1000, 
                step = 0.05,
                key = 'rad_cut_dist' 
            )
            rad_stride = st.slider(
                label="Gap between normals (Pixels)" , 
                min_value = 1 , 
                max_value = 5 , 
                step = 1,
                value = 2,
                key = "stride"
            )
            create_rad_btn = st.form_submit_button(
                label="Compute Radial Profile" , 
                width = 'stretch', 
                type = 'primary',
                )
            if create_rad_btn :#or (st.session_state.skeleton_map is not None):
                _execute_rad_prof_creation()

        if st.session_state.radprof is not None:  
            if st.session_state.radprof.prof_dict is not None:
                tangent_fig = plot_radials(st.session_state.radprof)
                rad_prof_window.pyplot(tangent_fig)
        else :
            rad_prof_window.info("Click **Compute Radial Profile** button to continue")
            # rad_prof_window.info()
        # st.caption('''Fig.4 : Skeleton Local normals for extracting radial profiles''')
    else:
        tangent_plot.info("Click **Run Model** in `Trace Filament Crest` to continue")




with props_plot:
    if st.session_state.radprof is not None:  
        if st.session_state.radprof.prof_dict is not None:
            st.markdown("#### `4. Cloud Beam-Level Properties`")
            st.markdown("<span style='color:#00e5e5;'>local normals beam--grouping :material/arrow_forward_ios: Median beam-grouped radial profile :material/arrow_forward_ios: Compute $R_{bg}$ :material/arrow_forward_ios: Fit Plummer profies :material/arrow_forward_ios: Beam-level properties `CSV` table</span>", unsafe_allow_html = True, text_alignment='justify')

            prop_controls_loc , props_window =  st.columns([2,2],  
                                                           border = False)
            props_window = props_window.container()
            prop_controls = prop_controls_loc.expander(
                "Compute Properties" , 
                expanded=True
            )
            # prop_controls = st.form("Group radial profile")
            with prop_controls.form("rad_prof_group_form"):
                beam_stride = st.slider(
                    label="Beam Grouping Size (beams)", 
                    min_value = 0.5 , 
                    max_value = 2.0 , 
                    step = 0.5 , 
                    value = 1.0 , 
                    key = "beam_stride"
                )
                grop_prof_btn  = st.form_submit_button(
                    label="Create Beam Groups", 
                    width='stretch'
                )
                if grop_prof_btn :#or st.session_state.radprof is not None:
                    with st.spinner("Grouping Profiles"):
                        st.session_state.radprof.group_profiles(
                            stride = st.session_state.beam_stride
                        )
                        st.session_state.radprof.beamProps = None

            if st.session_state.radprof.beam_dict is not None:
                
                # prop_controls.info('Now we can compute beam level properties')
                compute_prop_btn = prop_controls.button(
                    label="Compute Properties",
                    width='stretch',
                    type = 'primary'
                   
                )
                if compute_prop_btn :#or st.session_state.radprof.beam_dict is not None: 
                    with filament_beam_plot.spinner("Computing Properties"):
                        st.session_state.radprof.get_all_beam_props()
                # if prop_controls.button('Plot', width='stretch'):
                st.session_state.beam_prop_fig = plot_beam_groups(
                        st.session_state.radprof,
                        # sizescale = 10,
                        # figsize = (6,10),
                        )
                if st.session_state.beam_prop_fig is not None:
                    props_window.pyplot(st.session_state.beam_prop_fig,
                            width = 'stretch',
                        )

                # prop_controls_loc.markdown('''> Figure : Radial profile trace perpendicular to beam-grouped filament profiles''')
            else : props_window.info("Run **Create beam Groups** to continue")
        else:
            st.info("Run **Compute Radial Profile** in `Compute Radial Profile` to continue")

@st.fragment
def filament_beam_analysis(findex_list):
    fil_window , beam_window = st.columns([1,1], border=False)
    with fil_window.container(border=HEADING_BORDER):
        st.markdown("#### `5.Individual Filament Analysis`")
        st.markdown("<span style='color:#00e5e5;'>"+r""" Beams filtered by Plummer-fit reduced $\chi^2$ :material/arrow_forward_ios: Valid beams rejoined via MST"""+"</span>", unsafe_allow_html = True, text_alignment='justify')
    
    with beam_window.container(border=HEADING_BORDER):
        st.markdown("#### `6. Beam Level Analysis`")
        st.markdown('''<span style='color:#00e5e5;'>Seperate Plummer fit on each side, only till $R_{bg}$ (slope of profile $\sim$ 0):</span><br> '''+r"<span style='font-size:0.8rem;'>$N(H_2)(r)=N(H_2)^{bg}+N(H_2)^0\,\left[1+(r/R_{flat})^2\right]^{-(p-1)/2}$</span>", unsafe_allow_html = True, text_alignment='justify') 
        
    def increment():
        if st.session_state.selected_findx < len(findex_list):
            st.session_state.selected_findx += 1
    def decrement():
        if st.session_state.selected_findx > 0:
            st.session_state.selected_findx -= 1
    prev , slider, next = fil_window.columns([1,3,1], vertical_alignment = 'center')
    filindx = slider.select_slider(
        "Select Filament Index",
        options=findex_list,
        value=11,
        key="selected_findx"
    )
    with prev:
        st.button(":material/arrow_back_ios: Previous", on_click=decrement, key = 'f-') 
    with next:
        st.button("Next:material/arrow_forward_ios:", on_click=increment, key='f+')
    ax = st.session_state.radprof.plot_filament(
        findx=filindx,
        sizescale=20,
        show_beamid=True,
        red_chi_filter=10,
        contrast_filter=0.,
    )
    fil_window.pyplot(plt.gcf(), width=600)

    fil_df = st.session_state.radprof.beamProps.loc[filindx, :, :]
    beamindex_list = fil_df.reset_index()['beamIndex'].unique()
    if len(beamindex_list) > 1:
        # b_prev , b_slider, b_next = beam_window.columns([1,5,1], vertical_alignment = 'center')

        beamindx = beam_window.select_slider(
            "Select Beam Index",
            options=beamindex_list,
            value=beamindex_list[0],
            key="selected_beam"
        )
    else:
        beamindx = beamindex_list[0]
    
    beam_fig = plot_selected_beam(
        st.session_state.radprof.beam_dict['beam_elements'][beamindx]
    )
    beam_window.pyplot(beam_fig)
    # beam_window.caption("Fig.6: Plummer-fit on Median Radial profile of selected beam.")


# call
with filament_beam_plot:
    if st.session_state.radprof is not None:
        if st.session_state.radprof.beamProps is not None:
            findex_list = st.session_state.radprof.beamProps.reset_index()['filID'].unique()
            filament_beam_analysis(findex_list)
        else:
            st.info("Properties not computed yet. Run **Compute Properties** inside `Cloud Beam-Level Properties` to inspect individual filaments and beams")


with st.container(border=True):
    if st.session_state.radprof is not None:
        if st.session_state.radprof.beamProps is not None:
            st.dataframe(st.session_state.radprof.beamProps)
        else: st.info("Properties not computed yet. Run **Compute Properties** inside `Cloud Beam-Level Properties` to inspect individual filaments and beams")
    else: 
        st.write("Radial profiles not created yet. Run **create beam groups** to continue")



val_row = st.container(width='stretch', border = False)

with val_row:
    val_row.subheader(':primary[:material/design_services: Comparative Study] `On Synthetic Cloud `')
    val_intro , synth_fig, comp_curves, comp_fig  = st.columns([0.9,1,1.5,0.9], border = True, gap='small', vertical_alignment="top")
    with val_intro:
        st.markdown('''
Synthetic Plummer filaments are embedded in fBm turbulent backgrounds[5] of increasing amplitude. Sütra maintains precision and recall of ~0.98–0.80 across all levels, while FilFinder[6] and DisPerSE[1] degrade to precision ~0.2 requiring parameter retuning, demonstrating Sütra's robustness without any adjustment.
                    ''', text_alignment='justify')
    with synth_fig:

# use_case_image.image(BASE_DIR/'poster_images'/'monr2-filament-example-2.png', width=430)
        
        st.image(BASE_DIR/'poster_images'/'SYNTH-CD-1.png', width= 600 )
        st.caption("Fig.7: Synthetic filament and background using fBm")
    with comp_fig:
        st.image(BASE_DIR/'poster_images'/'SYNTH-CDcomp.png', width = 550)
    with comp_curves:
        st.image(BASE_DIR/'poster_images'/'SYNTH-CD-score.png', width=800)
        st.caption('Fig.8(a):Comparison of Sutra with FilFinder and DisPerSE, Fig.8(b) visualisation for k=1 and 3')


# st.subheader(':primary[:material/design_services: Application on the Galactic Plane]')

# from streamlit_extras.stylable_container import stylable_container as sc

applicaion_container = st.container(width='stretch' , border = False) 
with applicaion_container:
    applicaion_container.subheader(':primary[:material/design_services: Application on The Galactic Plane] `Hi-GAL Survey`')
    desc_text, mosaic_plot , scatter_plot = st.columns([2,6,3], border=True, vertical_alignment='top',)


    desc_text.markdown("Hi-GAL maps[7] a $2^o$-wide strip across the Galactic plane. Applying Sütra to a Hi-GAL mosaic, filament bulk properties are derived at beam resolution. The variation of linear mass density with peak column density illustrates the range of physical conditions sampled. The offset between filament-level and beam-level distributions quantifies density variation along individual axes, highlighting the value of beam-scale measurements over median-profile approaches.", text_alignment='justify')

    
    mosaic_plot.image(BASE_DIR/'poster_images'/'higal-tile.png', width='stretch')
    mosaic_plot.caption('Fig.9: Sütra skeletons overlaid on a Hi-GAL column density cutout[7].')
    scatter_plot.image(BASE_DIR/'poster_images'/'corr-plot.png', width = 'stretch')
    scatter_plot.caption('Fig.10:LMD vs. peak CD at filament and beam-level.')


st.divider()
footer = st.container(width='stretch', border=False)
footer_left , footer_right = footer.columns([3,2], border=False)
with footer_left:
    st.markdown("**Sutra availability**: Sutra is available as Python package and can be used with Streamlit based web UI, from the terminal as a CLI application and as a standard python package. Detailed paper on Sutra framework:  _Sutra:An integrated framework for filament identification and Characterisation_ , Kumaran et.al (under review). Development version @ : https://github.com/KumaranShivam5/sutra ; scan the QR code.")
with footer_right:
    st.markdown('''**References** : [1]Sousbie T., 2011, MNRAS; [2]Men’shchikov, A. 2020, ASCL; [3]Arzoumanian D., et al., 2019, A&A; [4]Motte F., et al., 2010a, A&A; [5]Robitaille, J.-F. et al. 2020, A&A, [6]Koch E. W., Rosolowsky E. W., 2015, MNRAS [7] Molinari S., et al., 2010, A&A 
                ''')


