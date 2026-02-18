

# --------------------------------------------------------------
# Initialise the logger **once** for the current user session.
# This must be done *after* `import streamlit as st`.
# --------------------------------------------------------------

# import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from sutra.file_io import load_fits
from sutra.filament import (
    run_identification,
)
from astropy.wcs import WCS
import streamlit as st

from sutra.plots import make_wcs_good
from sutra.tracer.modifiers import remove_nan
from sutra.measurement import skeleton ,apply_skl_threshold, local_field
from sutra.plots import plot_skeleton , plot_props_map_plotly


# from sutra.profiler.Filament import propsMap
from sutra.plots import plot_props_map


from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent 

DATA_PATH = str(BASE_DIR / "data" )
# print('--------------------------------')
# print(str(PRED_PATH))


# import sys
# from streamlit.web import cli as stcli
# def main():
#     sys.argv = ["streamlit", "run", __file__]
#     sys.exit(stcli.main())

# if __name__ == "__main__":
#     main()


def handle_filament_selection():
    event = st.session_state.all_beams
    if event and event.selection.points:
        selected_fil = event.selection.points[0]['customdata']
        st.session_state.selected_filament_index = selected_fil
        st.session_state.selected_beam_index = 0
    else : st.session_state.selected_filament_index = None


# from utils.streamlit_logger import _init_global_logger


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

st.set_page_config(
    page_title="SUTRA -Filament Tools",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# 1️⃣  Initialise session state (only once)
# ----------------------------------------------------------------------
if "cd_map" not in st.session_state:
    st.session_state["cd_map"] = load_fits(f"{DATA_PATH}/taurus-test.fits")         # will hold the 2‑D array
    # st.session_state["cd_map"] = None      # will hold the 2‑D array
if "prob_map" not in st.session_state:
    st.session_state["prob_map"] = None       # will hold the binary map
if "skeleton" not in st.session_state:
    st.session_state["skeleton"] = None   # optional, for the 3rd column

if "field" not in st.session_state:
    st.session_state["field"] = None   # optional, for the 3rd column

if "fil_table" not in st.session_state:
    st.session_state["fil_table"] = None   # optional, for the 3rd column

if "skl_fig" not in st.session_state:
    st.session_state["skl_fig"] = None   # optional, for the 3rd column


if "props_map_table" not in st.session_state:
    st.session_state["props_map_table"] = None   # optional, for the 3rd column


if "meta_info" not in st.session_state:
    st.session_state["meta_info"] = None   # optional, for the 3rd column


if "selected_filament_index" not in st.session_state:
    st.session_state.selected_filament_index = None



if "selected_beam_index" not in st.session_state:
    st.session_state.selected_beam_index = None


st.session_state.setdefault("expand_upload", True)
st.session_state.setdefault("expand_model", False)
st.session_state.setdefault("expand_skl", False)

# ----------------------------------------------------------------------
# 2️⃣  Sidebar widgets
# ----------------------------------------------------------------------


# skl_ex = st.expander("Skeletonization", expanded = True)



# --------------------------------------------------------z
# MAIN SECTION LAYOUT
# --------------------------------------------------------

st.title("Sutra: `ISM filament identification and Characterisation`")


st.sidebar.container()
st.sidebar.subheader('ISM FIlament :  Molecular cloud column density map')

identification  = st.container(border = False, )
# identification.markdown("**Identification**: `From Column Dnesity map to filament skeleton`")




characterisation = st.container(border = False)
# characterisation.markdown("**Characterisation** : `Measure filament physical properties`")




cdview_wrap , probview_wrap  = identification.columns([1,1], border=True)

cdview_wrap.markdown('### `Input Column Density Map`')
probview_wrap.markdown('### `Column Density map to Skeleton map`')

props_view , props_table = characterisation.columns([1,1.5], border = True)
props_view.write("### `Radial Profiles`")
props_table.write("### `Filament Physical Properties Catalogue`")

inp_ex , cdview = cdview_wrap.columns([1,2.5])
# inp_ex , cdview = cdview_wrap.columns([1,2])
# cdview = cdview_wrap.container()
skl_ctrl, probview = probview_wrap.columns([1,2.5])

# inp_ex = st.sidebar.expander('Input: Column-Density Map', expanded = st.session_state["expand_upload"])

uploaded_file = st.sidebar.file_uploader(
    "Upload Column Density map (FITS)", type=["fits" , "gz"], accept_multiple_files=False
)

# ----------------------------------------------------------------------
# 3️⃣  FILE UPLOAD
# ----------------------------------------------------------------------

# meta_info_block = st.sidebar.expander('Observation Parameters', expanded =True)
# if st.session_state['meta_info'] is None:
#     # meta_info = {
#     #         'distance' : 140, # 140 pc for Taurus , 260 for musca
#     #         'beam' : 36.3 , # arcsec 
#     #         # 'radial-cutoff' : meta_info_block.number_input("Radial profile cutoff (pc)", value = 0.2)# 0.2 pc , 
#     #         'radial-cutoff' : 0.2 , 
#     #         # 'prob-thresh' :
#     #     }
#     meta_info = {
#                 'distance' : meta_info_block.number_input("Distance (pc)", min_value = 10, value = 140), # 140 pc for Taurus , 260 for musca
#                 'beam' : meta_info_block.number_input("Beam Size (arcsec)", value = 36.3) , # arcsec 
#                 # 'radial-cutoff' : meta_info_block.number_input("Radial profile cutoff (pc)", value = 0.2)# 0.2 pc , 
#                 'radial-cutoff' : 0.2 , 
#                 # 'prob-thresh' :
#             }
#     st.session_state["meta_info"] = meta_info
# else: meta_info = st.session_state['meta_info']




if uploaded_file is not None:
    st.session_state["expand_model"] = True
    # model_ex = inp_ex.expander("Generate Filament Probability map", expanded = st.session_state["expand_model"])
    model_ex = inp_ex.container(border=True)
    
    # meta_info_block = inp_ex.expander('Observation Parameters', expanded =True)
    # meta_info_block = inp_ex.expander('Observation Parameters', expanded =True)
    meta_info = {
                'distance' : model_ex.number_input("Cloud Distance (pc)", min_value = 10, value = 140), # 140 pc for Taurus , 260 for musca
                'beam' : model_ex.number_input("Beam Size (arcsec)", value = 36.3) , # arcsec 
                # 'radial-cutoff' : meta_info_block.number_input("Radial profile cutoff (pc)", value = 0.2)# 0.2 pc , 
                'radial-cutoff' : 0.2 , 
                # 'prob-thresh' :
            }
    model_option = model_ex.selectbox(
        "Select ML model",
        options=["HGBS", "HiGAL", "Other"],
        index=0,
    )
    run_model_btn = model_ex.button("Run Model", type='primary', use_container_width = True)
else: st.sidebar.info("Upload Column Density map to continue")


# ----------------------------------------------------------------------
# 3️⃣  Load the FITS file (only when a file is present)
# ----------------------------------------------------------------------
if uploaded_file is not None:
    cd_array = load_fits(uploaded_file)
    cd_array.data = remove_nan(cd_array.data)  
    log_cd = cd_array.data
    log_cd[log_cd<=0.0] = np.nan
    log_cd = np.log10(log_cd)
    cd_header = cd_array.header             # <-- returns a np.ndarray
    st.session_state["cd_map"] = cd_array             # store for later pages

    # ---- left column: CD map -------------------------------------------------
    with cdview:
        # st.write("Input: Column-Density Map")
        # fig, ax = plt.subplots(figsize = (3,3), subplots_kws ={ "projection": WCS(cd_header)})
        fig = plt.figure(figsize=(3.8,3.8))
        ax = fig.add_subplot(111, projection=WCS(cd_header))
        
        # Apply the colour‑range chosen by the user
        # cdview = st.container(border=False)
        # cd_cmap_min, cd_cmap_max = st.expander('Adjust Colormap', expanded=False).slider(
        #         "Column Density map : colour range (min, max) :  $Log(N_{H_2}) $",
        #         min_value = 18.0,
        #         max_value = float(np.nanmax(log_cd)),
        #         value=(float(np.nanmin(log_cd)), float(np.nanmax(log_cd))),
        #         step = 0.5,
        #     )
        # vmin, vmax = cd_cmap_min, cd_cmap_max
        vmin, vmax = float(np.nanmin(log_cd)), float(np.nanmax(log_cd))
        ax.imshow(log_cd, cmap="inferno", origin="lower", vmin=vmin, vmax=vmax)
        ax = make_wcs_good(ax)
        # ax.set_axis_off()
        cdview.pyplot(fig, use_container_width=True)
else:
    with cdview_wrap:
        st.info("Upload **Column Denisty map** to continue")


# ----------------------------------------------------------------------
# 4️⃣  Compute the skeleton **only when the button is pressed**
# ----------------------------------------------------------------------
if uploaded_file is not None:
    if run_model_btn:
        st.session_state["skeleton"] = None
        st.session_state["field"] = None
        st.session_state["fil_table"] = None
        st.session_state["props_map_table"] = None
        st.session_state["skl_fig"] = None
        st.session_state["expand_upload"] = False
        st.session_state["expand_model"] = False
        
        with probview:
            with st.spinner(f"Running the {model_option} model"):
        # with redirect_stdout(writer), redirect_stderr(writer):
                prob_map = run_identification(cd_array.data, model_option, batch_size = None, )
        # Store the result so that it survives any later interaction
        st.session_state["prob_map"] = prob_map

# ----------------------------------------------------------------------
# 5️⃣  Show the skeleton (only if it exists in session_state)
# ----------------------------------------------------------------------

import matplotlib.colors as clr
cm_pl = clr.LinearSegmentedColormap.from_list('bin',[(0,(0,0,0,0)), (1,(1,0.3,0.3,1))])



# def plot_skeleton_over_prob_map(skl, ax):



from sutra.plots import compute_skeleton_image
if st.session_state["prob_map"] is not None:
    with probview:
        # probview_wrap.write("Model Output: Filament Probability map")
        fig = plt.figure(figsize=(3.8,3.8))
        cd_array =  st.session_state["cd_map"]
        ax = fig.add_subplot(111, projection=WCS(cd_array.header))
        prob_view = st.container(border=False)
        vmin, vmax = 0.05, 0.9
        ax.imshow(st.session_state["prob_map"], cmap="Grays_r", origin="lower", vmin=vmin, vmax=vmax)
        ax = make_wcs_good(ax)
        if st.session_state["skeleton"] is not None:
            skl_dialated , _ = compute_skeleton_image(st.session_state["skeleton"], _header = None)
            ax.imshow(skl_dialated > 0 , cmap=cm_pl)
        prob_view.pyplot(fig, use_container_width=False)
        # meta_info
        field = local_field(st.session_state["cd_map"], st.session_state["prob_map"] , meta_info = meta_info)
        st.session_state["field"] = field
else:
    with probview:
        st.info("**Run Model** after uploading CD map")


# ----------------------------------------------------------------------
# 5️⃣  Probability Map to Skeleton
# ----------------------------------------------------------------------

from sutra.file_io import download_fits
if st.session_state["field"] is not None:
    # probview.write('Skeletonize : Filament Skeleton Map')
    skl_viewport = probview.container(border=False)
    skeleton_controls = skl_ctrl.expander("Skeletonize", expanded = True)
    prob_thresh = skeleton_controls.slider("Probability Threshold (pc)", value = 0.4 , min_value = 0.1,max_value = 0.8, step = 0.05)
    make_skeleton = skeleton_controls.button('Update Skeleton', use_container_width = True, type = 'primary')
    if make_skeleton and st.session_state["field"] is not None:
        field = st.session_state["field"]
        field.apply_skl_threshold(prob_thresh)
        st.session_state["skeleton"] = field.field.skel
        st.session_state["field"] = field

    if(st.session_state["skeleton"] is not None) :
        # button_cols.download_button('All', download_df(current_df), file_name = 'csc-classification.csv', use_container_width=True, icon = ":material/download:")
        fits_buffer =  download_fits(data=np.asarray(st.session_state['skeleton'], dtype='int'), _header = cd_header)
        down_skel_btn = skeleton_controls.download_button("Downoad Skeleton", data = fits_buffer.getvalue() , use_container_width=True, icon = ":material/download:", file_name = 'skeleton.fits', mime = "application/fits")
        # st.rerun()
        # skl_fig = plot_skeleton(field.field.skel , _header = st.session_state["cd_map"].header, cd = st.session_state["cd_map"].data)
        # st.session_state['skl_fig'] = skl_fig
    
    # if st.session_state["skl_fig"] is not None:
        # skl_viewport.pyplot(st.session_state["skl_fig"])
    # else: probview.info("Run Extraction")
else: probview.info("Run Skeletonization")



# ----------------------------------------------------------------------
# 3️⃣  RADIAL PROFILE COMPUTATION
# ----------------------------------------------------------------------

from sutra.plots import plot_props_map_plotly_v2
# with ext_op:
if st.session_state["skeleton"] is not None:

    props_view_controls , props_view_display = props_view.columns([1,1.5])
    ext_params = props_view_controls.container(border = True)
    plot_control  = props_view_controls.container()
    plot_show_box = props_view_display.container()
    # ks = ext_params.number_input("Tangent Line Length (pix)", value = 7 , min_value = 5, max_value = 10,)
    # stride = ext_params.number_input("Radial Cut strides (pix)", value = 3, max_value = ks-2)
    # reorder = ext_params.number_input("Assemble Distance (pix)", value = 10, min_value = stride, max_value = 15)
    # props_stride = ext_params.number_input("Properties_stride (beam)", value = 1.0, min_value = 0.5, max_value = 3.0)

    # meta_info_block = ext_params.expander("Observation Parameters", expanded=True)
    # meta_info = {
    #         'distance' : ext_params.number_input("Distance (pc)", min_value = 10, value = 140), # 140 pc for Taurus , 260 for musca
    #         'beam' : ext_params.number_input("Beam Size (arcsec)", value = 36.3) , # arcsec 
    #         # 'radial-cutoff' : meta_info_block.number_input("Radial profile cutoff (pc)", value = 0.2)# 0.2 pc , 
    #         'radial-cutoff' : 0.2 , 
    #         # 'prob-thresh' :
    #     }
    props_stride = ext_params.slider("Measurment stride (beam)", value = 0.5 , min_value = 1.0,max_value = 3.0, step = 0.5)
    # st.session_state["meta_info"] = meta_info
    
    ext_porf_btn = ext_params.button("Extract Radial profiles", type = 'secondary')
        
    if ext_porf_btn and st.session_state["field"] is not None:
        field = st.session_state["field"]
        field.field.meta_info = meta_info
        if st.session_state["fil_table"] is None:
            with st.spinner("Computing physical properties of all skeletons"):
                print("[INFO] >>> Extracting Radial profiles")
                field.process_skeleton(ks = 5, stride = 2 , reorder = 15)
                field.extract_rad_prof()
                musca_fil_table =  field.get_filament_table()
                musca_fil_table = musca_fil_table.drop(columns = ['Location']) 
                st.session_state["fil_table"] = musca_fil_table
        props_df = field.get_filament_prop_map(stride = 0.5 , refresh=True)
        props_df['rad_pix']  = field.field.pc_to_pixel(props_df['W_bg']/4)
        st.session_state["props_map_table"] = props_df
        st.session_state["field"] = field
    
    if st.session_state["props_map_table"] is not None:
        prop_tab = st.session_state["props_map_table"]

        # size_btn , color_btn  = plot_control.columns([1,1])
        color_by = plot_control.selectbox("Color By", options = prop_tab.columns.to_list(), index = 2)
        size_by = plot_control.selectbox("Size By", options = prop_tab.columns.to_list(), index =  3)
        zmin,zmax = plot_control.slider("Colormap : ", min_value = 18, max_value=24, value=(20,22))

        props_fig = plot_props_map_plotly_v2(st.session_state["cd_map"].data, df = prop_tab , size_by = size_by, color_by = color_by, skeleton=st.session_state["skeleton"], zmin=zmin, zmax = zmax)
        selected_beams = plot_show_box.plotly_chart(props_fig, selection_mode = 'points', on_select = handle_filament_selection, key = "all_beams" )
        # selected_beams
        selected_points = selected_beams.selection["point_indices"]
        
        if len(selected_points)>0:
            selected_filament = selected_beams['selection']['points'][0]['customdata'][0]
            # selected_filament
            st.session_state.selected_filament_state = selected_filament
    else: props_view_display.info("Run Properties extraction")

else: props_view.info("Convert prob-map to skeleton for extrating radial profiles")



# ----------------------------------------------------------------------
# 3️⃣  INSPECTING PROFILES
# ----------------------------------------------------------------------

from sutra.plots import plot_onefil_props, plot_onefil_props_plotly




def handle_beam_selection():
    event = st.session_state.selected_beam
    if event and event.selection.points:
        selected_beam = event.selection.points[0]['point_index']
        st.session_state.selected_beam_index = selected_beam
    else : st.session_state.selected_beam_index = None






if st.session_state["fil_table"] is not None:
    filament_display, props_df_display   = props_table.tabs(["Inspect individual filament", "Properties Table of all filaments", ])
    props_df_display.dataframe(st.session_state["fil_table"])

    with filament_display:
     
        if st.session_state.selected_filament_index is not None:
            findx = st.session_state.selected_filament_index
            df = st.session_state["props_map_table"]
            df_fil = st.session_state["fil_table"]
            filprop , filmap,  beam_profile_view  = st.columns([1,2,3]) ## LAYOUT 
            selected_fil_table  = df_fil.loc[findx[0]]
            selected_fil_table.index.name = f'Fil-ID : '
            # selected_fil_table.rename(columns = {'Findex' : 'FilID'})
            filprop.write(
                selected_fil_table
            )
            # st.write("f-index", findx[0])
            df_tmp = df[df['filID'] == findx[0]]
            if(len(df_tmp)>3):
                one_fil_fig = plot_onefil_props_plotly(st.session_state["cd_map"].data , df_tmp, crop_map = True, size_by = 'W_bg')
                beam_selection = filmap.plotly_chart(one_fil_fig, key = "selected_beam", selection_mode = 'points', on_select=handle_beam_selection )
                # beam_profile_view.write(beam_selection.selection)
            else:
                st.error("The filament does not have more than 3 beams satisfying the Plummer fit")
        else: st.info('No filament selected')

else: 
    props_table.info("Filament properties not computed")

# from .profiler.Filament import Field
# @st.cache_resource
# def plot_profiles(findx , bindx):
#     cloud = st.session_state['field'].filament_collection
#     cloud.collect_filaments()
#     fil = cloud.filaments[findx]

from sutra.profiler.PlotFil import plot_radial_profile
if st.session_state.selected_beam_index is not None and st.session_state.selected_filament_index is not None and st.session_state["fil_table"] is not None:
    findx = st.session_state.selected_filament_index[0]
    bindx = st.session_state.selected_beam_index
    beam_profile_view.write(f'Selected Filament ID : {findx}\n Selected beam ID : {bindx}')
    fig = plot_radial_profile(st.session_state['field'].filament_collection.filaments[findx],  pl_indx = bindx, getfig = True)
    beam_profile_view.pyplot(fig)