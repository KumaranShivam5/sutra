# ──────────────────────────────────────────────────────────────────────────────
#   SUTRA – Filament tools (cleaned Streamlit implementation)
#   Core scientific functions are unchanged – only the Streamlit plumbing
#   has been reorganised.
# ──────────────────────────────────────────────────────────────────────────────

# -------------------------------------------------
# 1️⃣  Imports & tiny helpers
# -------------------------------------------------
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from astropy.wcs import WCS

# ---- SUTRA library -------------------------------------------------
from sutra.file_io import load_fits, download_fits



from sutra.tracer.modifiers import remove_nan
from sutra.measurement import local_field as LocalField   # <-- rename to avoid shadowing
from sutra.plots import (
    make_wcs_good,
    compute_skeleton_image,
    plot_props_map_plotly_v2,
    plot_onefil_props_plotly,
    # plot_radial_profile,
)

from sutra.profiler.PlotFil import plot_radial_profile
from sutra.tracer.predictor import filamentIdentifier as FID

from sutra.plots import plot_props_map   # keep original import (unused in cleaned UI)
# -------------------------------------------------
# 2️⃣  Cache heavy pure‑python functions
# -------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_fits_cached(fobj):
    """Read a FITS file (path‑like or UploadedFile) and return a `fits` object."""
    return load_fits(fobj)

# @st.cache_data(show_spinner=False)
# def _run_model(data, model_name):
#     """Run the chosen ML model on the column‑density array."""
#     return run_identification(data, model_name, batch_size=None)

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

# -------------------------------------------------
# 4️⃣  Small UI helpers (selection callbacks)
# -------------------------------------------------
def _handle_filament_selection():
    """Callback for the map‑plot selection – stores the chosen filament."""
    ev = st.session_state.get("all_beams")
    if ev and ev.selection and ev.selection.get("points"):
        sel_pt = ev.selection["points"][0]
        # customdata is a list → we store the whole tuple so the downstream code
        # can keep the same logic (`[0]` is the filament ID)
        st.session_state.selected_filament_index = sel_pt["customdata"]
    else:
        st.session_state.selected_filament_index = None


def _handle_beam_selection():
    """Callback for the beam‑selection plot – stores the chosen beam."""
    ev = st.session_state.get("selected_beam")
    if ev and ev.selection and ev.selection.get("points"):
        st.session_state.selected_beam_index = ev.selection["points"][0]["point_index"]
    else:
        st.session_state.selected_beam_index = None


# -------------------------------------------------
# 5️⃣  Page layout – defined *after* all state changes
# -------------------------------------------------



st.markdown("""
        <style>
        /* Example: Reduce top padding of the main content block */
        .block-container {
            padding-top: 2rem; /* Adjust as needed */
        }
        .stMainBlockContainer{
            padding-left: 4rem !important;
            padding-right : 4rem !important;
            padding-top:1rem !important;
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
        
        </style>
    """, unsafe_allow_html=True)

st.markdown(f"<h2> <span style = 'color:white;font-weight:normal !important;font-size:4rem;'>Sutra</span> <span style = 'color:#fbd48a;font-weight:normal;'> : Filaments in the Interstellar Medium</span></h2> ", unsafe_allow_html=True)

plt.style.use('dark_background')

st.set_page_config(

    page_title="SUTRA - Filament tools",
    layout="wide",
    initial_sidebar_state="expanded",
)



st.sidebar.container()
st.sidebar.subheader('ISM FIlament :  Molecular cloud column density map')

# ── 5.1️⃣  Sidebar – upload + model launch ─────────────────────────────────────
st.sidebar.title("📂  Column‑density map")
uploaded_cd = st.sidebar.file_uploader(
    "Upload a FITS file containing the column‑density map",
    type=["fits"],
    key="upload_cd",
)

if uploaded_cd is not None:
    # ---- Load CD map (cached) ----------------------------------------------
    cd_fits = _load_fits_cached(uploaded_cd)
    cd_fits.data = remove_nan(cd_fits.data)                # keep original behaviour
    st.session_state.cd_map = cd_fits
    # Reset everything that depends on a new CD map
    # print('----resetting everything---')
    # st.session_state.prob_map = None
    # st.session_state.skeleton = None
    # st.session_state.local_field = None
    # st.session_state.fil_table = None
    # st.session_state.props_map_table = None
    # st.session_state.selected_filament_index = None
    # st.session_state.selected_beam_index = None

# ---- Meta‑info (distance, beam, …) – always stored in session_state ----------
# if st.session_state.cd_map is not None:
#     with st.sidebar.expander("🔧  Model parameters", expanded=True):
#         distance = st.number_input(
#             "Distance (pc)", min_value=1.0, value=100.0, step=1.0, key="meta_dist"
#         )
#         beam = st.number_input(
#             "Beam (pixel)", min_value=1.0, value=5.0, step=0.5, key="meta_beam"
#         )
#         # Store the dict *once* – it will be re‑used later without re‑creating it
#         st.session_state.meta_info = {"distance": distance, "beam": beam}

# -------------------------------------------------
# 5️⃣  Main layout – three columns that stay static
# -------------------------------------------------
st.subheader(':primary[:material/draw: Trace filament Skeleton]')

identification_window = st.container(border  = False , width = 'stretch')
st.subheader(':primary[:material/design_services: Measure Filament properties]')

characterisation = st.container(border = False , width='stretch')
cd_col, prob_col = identification_window.columns([1, 1], border=True)

rad_prof_col , tables_col = characterisation.columns([5,7], border = True) 
# ----------------------------------------------------------------------
#   LEFT column – upload / model
# ----------------------------------------------------------------------
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
        if st.session_state.skeleton is not None:
                skel_img, _ = _compute_skel_image(st.session_state.skeleton)
                ax.contour(skel_img,  colors="white", linewidths = 0.5, )
        ax = make_wcs_good(ax)
        cd_col_plot.pyplot(fig, use_container_width=True)

    # ---- Run model button ------------------------------------------------
    with cd_col_controls:
        if st.session_state.cd_map is not None:
            with st.expander("CD parameters", expanded=True):
                distance = st.number_input(
                "Distance (pc)", min_value=1.0, value=140.0, step=1.0, key="meta_dist"
                )
                beam = st.number_input(
                    "Beam (arcsec)", min_value=1.0, value=36.4, step=0.1, key="meta_beam"
                )
                # Store the dict *once* – it will be re‑used later without re‑creating it
                st.session_state.meta_info = {"distance": distance, "beam": beam}

            with st.expander("Model", expanded=True):
                model_name = st.selectbox(
                    "Choose ML model",
                    options=["HGBS", "HiGAL"],   # keep the names you actually support
                    key="model_selector",
                )


                run_btn = st.button(":material/batch_prediction: Run model", key="run_model_btn", type = 'primary', width =  'stretch')
                if run_btn:
                    with st.spinner("Running the model …"):
                        prob = FID(predictor_name=model_name).predict(st.session_state.cd_map.data)
                        st.session_state.prob_map = prob
                        # Reset everything that depends on a new prob‑map
                        st.session_state.skeleton = None
                        st.session_state.local_field = None
                        st.session_state.fil_table = None
                        st.session_state.props_map_table = None
                        st.rerun()
                reset_cd_btn = st.button(":material/restart_alt: Reset CD map", key = "reset_cd_btn" , type = 'secondary', width =  'stretch')
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
            ax.imshow(st.session_state["prob_map"], cmap="Greys_r", origin="lower", )
            ax = make_wcs_good(ax)
            # Overlay skeleton if it already exists
            if st.session_state.skeleton is not None:
                skel_img, _ = _compute_skel_image(st.session_state.skeleton)
                ax.contour(skel_img,  colors="red", linewidths = 0.3, alpha = 1)
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
            # st.write('ok instse if')
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

# ----------------------------------------------------------------------
#   RIGHT column – radial‑profile extraction & map visualisation
# ----------------------------------------------------------------------




with rad_prof_col:
    col_ctl, col_plot = st.columns([1, 3])
    col_ctl.markdown('### `3.Radial-profile`')


    # st.subheader("3️⃣  Radial‑profile extraction")
    if st.session_state.skeleton is None:
        st.info("Create a skeleton first (middle column).")
    else:
        # ---- Controls ----------------------------------------------------
        profile_ctrl = col_ctl.container()
        with profile_ctrl:
            stride = st.slider(
                "Measurement stride (beam)", min_value=1.0, max_value=3.0, step=0.5, value=0.5,
                key="stride_slider"
            )
            extract_btn = st.button(
                "Extract radial profiles", key="extract_profiles_btn", type="secondary"
            )

        if extract_btn:
            # -------------------------------------------------------------
            #   Heavy computation – keep it exactly as in the original code
            # -------------------------------------------------------------
            lf = st.session_state.local_field
            lf.radprof.meta_info = st.session_state.meta_info or {}
            if st.session_state.fil_table is None:
                with st.spinner("Computing physical properties of all skeletons"):
                    lf.process_skeleton(ks=5, stride=2, reorder = 5)
                    lf.extract_rad_prof()
                    fil_tab = lf.get_filament_table().drop(columns=["Location"])
                    st.session_state.fil_table = fil_tab

            # ---- Map of filament properties (used by the plot) ------------
            props_df = lf.get_filament_prop_map(stride=stride, refresh=True)
            # add pixel‑conversion column used by the plot
            props_df["rad_pix"] = lf.radprof.pc_to_pixel(props_df["W_bg"] / 4)
            st.session_state.props_map_table = props_df
            st.session_state.local_field = lf          # (object mutated in‑place)
            # st.rerun()

        # ------------------------------------------------------------------
        #   Visualise the property map (only when the dataframe exists)
        # ------------------------------------------------------------------
        if st.session_state.props_map_table is not None:
            df = st.session_state.props_map_table
            with st.container():
                
                with col_ctl:
                    colour_by = st.selectbox(
                        "Colour by", options=df.columns.tolist(), index=2, key="col_by_select"
                    )
                    size_by = st.selectbox(
                        "Size by", options=df.columns.tolist(), index=3, key="size_by_select"
                    )
                    zmin, zmax = st.slider(
                        "Colormap range (z)", min_value=15, max_value=30,
                        value=(20, 22), key="zrange_slider"
                    )
                with col_plot:
                    fig = plot_props_map_plotly_v2(
                        st.session_state.cd_map.data,
                        df=df,
                        size_by=size_by,
                        color_by=colour_by,
                        skeleton=st.session_state.skeleton,
                        zmin=zmin,
                        zmax=zmax,
                    )
                    sel = st.plotly_chart(
                        fig,
                        on_select=_handle_filament_selection,
                        selection_mode="points",
                        key="all_beams_plot",
                        width = "stretch", 
                        height = 600
                    )
                    # Store the last selected filament (if any)
                    if sel and sel.selection and sel.selection.get("points"):
                        pt = sel.selection["points"][0]
                        st.session_state.selected_filament_index = pt["customdata"]
                    else:
                        st.session_state.selected_filament_index = None
        else:
            st.info("Press *Extract radial profiles* to compute the map.")

# ----------------------------------------------------------------------
# 6️⃣  Inspection tabs – filament‑by‑filament & table view
# ----------------------------------------------------------------------
if st.session_state.fil_table is not None:
    tab_fil, tab_table = tables_col.tabs(["Inspect individual filament", "Properties table"])
    with tab_table:
        st.dataframe(st.session_state.fil_table)

    with tab_fil:
        if st.session_state.selected_filament_index is not None:
            fid = st.session_state.selected_filament_index[0]          # filament ID
            # ---- Show filament summary ------------------------------------------------
            col_prop, col_map, col_beam = st.columns([1, 2, 3])
            # – properties table ---------------------------------------------------------
            row = st.session_state.fil_table.loc[fid]
            row.index.name = "Fil‑ID"
            col_prop.write(row)

            # – map of the selected filament (radial‑profile map) ------------------------
            df_sel = st.session_state.props_map_table
            df_fil = df_sel[df_sel["filID"] == fid]

            if len(df_fil) > 3:
                # Plot the filament on the CD map (interactive Plotly)
                fig_one = plot_onefil_props_plotly(
                    st.session_state.cd_map.data, df_fil, crop_map=True, size_by="W_bg"
                )
                col_map.plotly_chart(fig_one, key="selected_beam_plot", selection_mode="points",
                                     on_select=_handle_beam_selection)
            else:
                st.error("The filament does not contain enough beams for a Plummer fit.")
        else:
            st.info("Select a filament on the map (right‑hand side).")
else:
    st.info("Run the model → skeleton → radial‑profile extraction to enable inspection.")

# ----------------------------------------------------------------------
# 7️⃣  Beam‑profile visualisation (when a beam & filament are selected)
# ----------------------------------------------------------------------
if (st.session_state.selected_filament_index is not None
        and st.session_state.selected_beam_index is not None
        and st.session_state.local_field is not None):
    fid = st.session_state.selected_filament_index[0]
    bid = st.session_state.selected_beam_index
    st.subheader(f"🔎  Filament {fid} – beam {bid}")

    fig = plot_radial_profile(
        st.session_state.local_field.filament_collection.filaments[fid],
        pl_indx=bid,
        getfig=True,
    )
    st.pyplot(fig)

# -------------------------------------------------
# 8️⃣  Selection‑handler utilities (must be defined *after* the widgets)
# -------------------------------------------------
def _handle_filament_selection():
    """Callback for the property‑map Plotly selection."""
    ev = st.session_state.get("all_beams_plot")
    if ev and ev.selection and ev.selection.get("points"):
        pt = ev.selection["points"][0]
        # customdata is a list → we keep the whole tuple (same shape as original code)
        st.session_state.selected_filament_index = pt["customdata"]
    else:
        st.session_state.selected_filament_index = None


def _handle_beam_selection():
    """Callback for the individual‑filament beam‑selection Plotly chart."""
    ev = st.session_state.get("selected_beam_plot")
    if ev and ev.selection and ev.selection.get("points"):
        st.session_state.selected_beam_index = ev.selection["points"][0]["point_index"]
    else:
        st.session_state.selected_beam_index = None

# -------------------------------------------------
# 9️⃣  Final tidy‑up – hide any stray “breakpoint” prints
# -------------------------------------------------
# (All the `st.write("Breakpoint …")` statements that were only for debugging
# have been removed – the UI now shows only the information the user really
# needs.)

# ──────────────────────────────────────────────────────────────────────────────
#   End of cleaned Streamlit app
# ──────────────────────────────────────────────────────────────────────────────