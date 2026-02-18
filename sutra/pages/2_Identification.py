# pages/2_🔎_Identification.py
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from astropy.wcs import WCS

from sutra.file_io import load_fits, download_fits

from sutra.file_io import load_fits, download_fits
from sutra.tracer.modifiers import remove_nan
from sutra.measurement import local_field as LocalField
from sutra.plots import (
    make_wcs_good,
    compute_skeleton_image,
)

from sutra.tracer.predictor import filamentIdentifier as FID

# from streamlitUtils import load_fits_cached, run_model, skeleton_image , download_fits , remove_nan , make_wcs_good

# from .streamlitUtils import *

# -------------------------------------------------
# Caching
# -------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_fits_cached(fobj):
    """Read a FITS file (path‑like or UploadedFile) and return a `fits` object."""
    return load_fits(fobj)

@st.cache_data(show_spinner=False)
def _run_model(data, model_name):
    """Run the chosen ML model on the column‑density array."""
    fid = FID(predictor_name=model_name)
    return fid.predict(data, batch_size=None)

@st.cache_data(show_spinner=False)
def _compute_skel_image(skel, header=None):
    """Return a binary image of the skeleton (used for overlay)."""
    return compute_skeleton_image(skel, _header=header)
# -------------------------------------------------
# 3️⃣  Session‑state initialisation (run once)
# -------------------------------------------------
def _init_session():
    """Create all keys that the app expects – only the first time."""
    defaults = {
        "cd_map": None,               # FITS object (column‑density)
        "prob_map": None,             # probability map (model output)
        "skeleton": None,             # binary skeleton image
        "local_field": None,          # LocalField object
        "fil_table": None,            # dataframe of filament properties
        "props_map_table": None,      # dataframe for map visualisation
        "meta_info": None,            # dict with distance, beam, … (saved in state)
        "selected_filament_index": None,
        "selected_beam_index": None,
        # UI‑expansion flags – they are read by the sidebar / expander widgets
        "expand_upload": True,
        "expand_model": False,
        "expand_skel": False,
        "expand_props": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_session()                     # executed on every script run, but only sets once




# st.markdown("""
#         <style>
#         /* Example: Reduce top padding of the main content block */
#         .block-container {
#             padding-top: 5rem; /* Adjust as needed */
#         }
#         .stMainBlockContainer{
#             padding-left: 4rem !important;
#             padding-right : 4rem !important;
#             padding-top:1rem !important;
#             }
#         .stAppHeader{ 
        
#             }
#         .stVerticalBlock{
#             gap:0.5rem !important;
#         }
#         .stSidebarHeader{
#             margin-bottom : 0 !important; 
#             height:0 !important;
#         }
#         /* Example: Reduce gap between elements in a specific column */
#         [data-testid="column"]:nth-of-type(1) [data-testid="stVerticalBlock"] {
#             gap: 2rem; /* Remove gap */
#         }
#         code {color:#fbd48a !important;}
        
#         </style>
#     """, unsafe_allow_html=True)



st.markdown("""
        <style>
        /* Example: Reduce top padding of the main content block */
       
        [data-testid="column"]:nth-of-type(1) [data-testid="stVerticalBlock"] {
            gap: 2rem; /* Remove gap */
        }
        code {color:#fbd48a !important;}
        
        </style>
    """, unsafe_allow_html=True)

plt.style.use('dark_background')




st.set_page_config(

    page_title="SUTRA - Filament tools",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Sutra: **`Filaments in the Interstellar Medium`**")


st.sidebar.container()
st.sidebar.subheader('ISM FIlament :  Molecular cloud column density map')
# ------------------------------------------------------------------
# Sidebar – CD map uploader (shared with other pages)
# ------------------------------------------------------------------
st.sidebar.subheader("📂 Column‑density map")
uploaded_cd = st.sidebar.file_uploader(
    "Upload a FITS file", type=["fits"], key="upload_cd"
)

if uploaded_cd is not None:
    cd_fits = _load_fits_cached(uploaded_cd)
    cd_fits.data = remove_nan(cd_fits.data)
    st.session_state.cd_map = cd_fits
    # reset downstream results
    # for k in ["prob_map", "skeleton", "local_field", "fil_table", "props_map_table"]:
        # st.session_state[k] = None

# ------------------------------------------------------------------
# Main columns
# ------------------------------------------------------------------
cd_col, prob_col = st.columns([1, 1], border=True)

# ---------- Left column – CD map preview & model launch ----------
with cd_col:
    st.markdown('### `1. Input Column Density Map`')

    cd_col_controls , cd_col_plot = st.columns([1,3])

    if st.session_state.cd_map is None:
        st.info("Upload a column‑density FITS file in the sidebar.")
    else:
        # Show a tiny preview of the CD map
        log_cd = st.session_state.cd_map.data
        log_cd[log_cd<=0.0] = np.nan
        log_cd = np.log10(log_cd)
        fig = plt.figure(figsize=(4,4))

        vmin, vmax = float(np.nanmin(log_cd)), float(np.nanmax(log_cd))
        ax = fig.add_subplot(111, projection=WCS(st.session_state.cd_map.header))
        ax.imshow(log_cd, cmap="inferno", origin="lower", vmin=vmin, vmax=vmax)
        ax = make_wcs_good(ax)
        cd_col_plot.pyplot(fig, use_container_width=True)

    # ---- Run model button ------------------------------------------------
    with cd_col_controls:
        if st.session_state.cd_map is not None:
            with st.expander("CD parameters", expanded=True):
                distance = st.number_input(
                "Distance (pc)", min_value=1.0, value=100.0, step=1.0, key="meta_dist"
                )
                beam = st.number_input(
                    "Beam (pixel)", min_value=1.0, value=5.0, step=0.5, key="meta_beam"
                )
                # Store the dict *once* – it will be re‑used later without re‑creating it
                st.session_state.meta_info = {"distance": distance, "beam": beam}

            with st.expander("Model", expanded=True):
                model_name = st.selectbox(
                    "Choose ML model",
                    options=["HGBS", "HiGAL", "Other"],   # keep the names you actually support
                    key="model_selector",
                )

                # btn_left , btn_right = st.columns([1,1], border = False, width =  'stretch')

            

                run_btn = st.button("Run model", key="run_model_btn", type = 'primary', width =  'stretch')
                if run_btn:
                    with st.spinner("Running the model …"):
                        prob = _run_model(st.session_state.cd_map.data, model_name)
                        st.session_state.prob_map = prob
                        # Reset everything that depends on a new prob‑map
                        st.session_state.skeleton = None
                        st.session_state.local_field = None
                        st.session_state.fil_table = None
                        st.session_state.props_map_table = None
                        st.rerun()

                reset_cd_btn = st.button("Reset CD map", key = "reset_cd_btn" , type = 'secondary', width =  'stretch')
                if reset_cd_btn:
                    st.session_state.cd_map  = None
                    st.session_state.prob_map = None
                    st.session_state.skeleton = None
                    st.session_state.local_field = None
                    st.session_state.fil_table = None
                    st.session_state.props_map_table = None
                    st.session_state.selected_filament_index = None
                    st.session_state.selected_beam_index = None
                    st.rerun()


# ----------------------------------------------------------------------
#   MIDDLE column – probability map → skeleton overlay
# ----------------------------------------------------------------------
with prob_col:
    st.markdown('### `2. Column Density map to Skeleton map`')
    if st.session_state.prob_map is None:
        print('-------BP 3----')
        st.info("Run the model (left column) to generate a probability map.")
    else:
        # ---- Show probability map with optional skeleton overlay ------------
        prob_controls , prob_view = st.columns([1,3])
        with prob_view:
            fig = plt.figure(figsize=(3.8,3.8))
            cd_array =  st.session_state["cd_map"]
            ax = fig.add_subplot(111, projection=WCS(cd_array.header))
            ax.imshow(st.session_state["prob_map"], cmap="Reds_r", origin="lower", )
            ax = make_wcs_good(ax)
            # Overlay skeleton if it already exists
            if st.session_state.skeleton is not None:
                skel_img, _ = _compute_skel_image(st.session_state.skeleton)
                ax.contour(skel_img,  colors="white", linewidths = 0.5, )
            st.pyplot(fig, use_container_width=True)

        # ---- Skeletonisation controls (expander) -------------------------
        skel_expander = prob_controls.expander(
            "Skeletonisation controls", expanded=True)
        with skel_expander:
            prob_thresh = st.slider(
                "Probability threshold (pc)",
                min_value=0.1,
                max_value=0.8,
                value=0.4,
                step=0.05,
                key="prob_thresh_slider",
            )
            update_btn = st.button(
                "Update skeleton", key="update_skel_btn", type="primary" , width = 'stretch'
            )

        # ------------------------------------------------------------------
        #   Create / update the LocalField object (only once, then read)
        # ------------------------------------------------------------------
        if st.session_state.local_field is None and st.session_state.prob_map is not None:
            # First time we need a LocalField instance → build it from the prob‑map
            lf = LocalField(st.session_state.cd_map, st.session_state.prob_map, meta_info=st.session_state.meta_info)
            st.session_state.local_field = lf
            print('--------- BP 1-----------')

        if update_btn and st.session_state.local_field is not None:
            print('--------- BP 2-----------')

            # -------------------------------------------------------------
            #   Apply new probability threshold → new skeleton + state update
            # -------------------------------------------------------------
            lf = st.session_state.local_field
            lf.apply_skl_threshold(prob_thresh)

            # store results
            st.session_state.skeleton = lf.radprof.skel
            st.session_state.local_field = lf
            st.write(st.session_state.skeleton)
            st.rerun()
            # force a rerun so the rest of the UI sees the new skeleton

            # ------------------------------------------------------------------
            #   Download button (only appears when a skeleton exists)
            # ------------------------------------------------------------------
        if st.session_state.skeleton is not None:
            st.write('ok instse if')
            header = st.session_state.cd_map.header if st.session_state.cd_map else None
            fits_buf = download_fits(
                data=np.asarray(st.session_state.skeleton, dtype="int"), _header=header
            )
            skel_expander.download_button(
                label="Download skeleton",
                data=fits_buf.getvalue(),
                file_name="skeleton.fits",
                mime="application/fits",
                # use_container_width=True,
                key="download_skel_btn",
                width = 'stretch'
            )
        else:
            st.info("Press *Update skeleton* to generate a binary skeleton.")
