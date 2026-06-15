import streamlit as st

from sutra.profilerV2.radprof import RadProf
from sutra.plot_utils import *



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

HEADING_BORDER = True

st.set_page_config(
    
    page_title="SUTRA - Filament tools",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.subheader(':primary[:material/design_services: Filament Characterisation  |  Beam-Level Measurements of Filament properties]')


tangent_plot , props_plot = st.columns([1,1], border=True)
filament_beam_plot = st.container(border=True)


if 'cd_map' not in st.session_state:
    st.info('Run Filament identification to continue with characterisation')




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
    st.markdown("### `3. Compute Radial Profile`")
    st.markdown("<span style='color:#00e5e5;'>Bezier curves skeleton smootheining :material/arrow_forward_ios: Local normnals computation :material/arrow_forward_ios: Radial Profiles extraction :material/arrow_forward_ios: Shift center of profiles to maxima</span>", unsafe_allow_html = True, text_alignment = 'justify')
    rad_controls , rad_prof_window = st.columns([1,2])
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
    fil_window_box , beam_window_box  = st.columns([1,1], border=True)
    fil_window , fil_props = fil_window_box.columns([3,1], border=True)
    beam_window , beam_props    = beam_window_box.columns([3,1], border=True)
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

