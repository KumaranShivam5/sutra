# pages/1_🏠_Home.py
import streamlit as st

st.set_page_config(page_title="SUTRA – Filament tools", layout="wide")
st.title("SUTRA – Filament identification & characterisation")

st.markdown(
    """
### What this app does
* **Identify** filamentary structures in a column‑density (CD) map using a trained ML model.  
* **Skeletonise** the probability map and extract radial profiles of each filament.  
* **Visualise** physical properties (width, contrast, …) and inspect individual filaments.

### How to use
1. **Identification** – go to *🔎 Identification* (page 2).  
2. **Characterisation** – after a skeleton appears, switch to *📊 Characterisation* (page 3).  

> The sidebar (present on every page) contains the CD‑map uploader and basic model parameters.
"""
)

# st.sidebar

def init_state():
    defaults = {
        "cd_map": None,
        "prob_map": None,
        "skeleton": None,
        "local_field": None,
        "fil_table": None,
        "props_map_table": None,
        "meta_info": None,
        "selected_filament_index": None,
        "selected_beam_index": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()