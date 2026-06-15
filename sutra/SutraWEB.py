import streamlit as st

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


