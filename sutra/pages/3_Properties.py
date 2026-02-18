# pages/3_📊_Characterisation.py
import streamlit as st
import numpy as np
from sutra.plots import (
    plot_props_map_plotly_v2,
    plot_onefil_props_plotly, 
)
# plt.style.use('dark_background')



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


st.set_page_config(page_title="Characterisation", layout="wide")
st.title("📊 Characterisation")

# ------------------------------------------------------------------
# Guard – make sure a skeleton exists
# ------------------------------------------------------------------
if st.session_state.skeleton is None:
    st.info("Create a skeleton first (Identification page).")
    st.stop()

# ------------------------------------------------------------------
# Column layout: radial‑profile controls | map & inspection tabs
# ------------------------------------------------------------------
rad_prof_col , tables_col = st.columns([1,2]) 

# ------------------- Radial‑profile extraction --------------------
with rad_prof_col:
    st.markdown('### `3. Radial-profile extraction`')

    # st.subheader("3️⃣  Radial‑profile extraction")
    if st.session_state.skeleton is None:
        st.info("Create a skeleton first (middle column).")
    else:
        # ---- Controls ----------------------------------------------------
        profile_ctrl = st.container()
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
                    lf.process_skeleton(ks=5, stride=2, reorder=15)
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
                col_ctl, col_plot = st.columns([1, 2])
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
                    )
                    # Store the last selected filament (if any)
                    if sel and sel.selection and sel.selection.get("points"):
                        pt = sel.selection["points"][0]
                        st.session_state.selected_filament_index = pt["customdata"]
                    else:
                        st.session_state.selected_filament_index = None
        else:
            st.info("Press *Extract radial profiles* to compute the map.")

# ------------------- Property‑map visualisation ------------------
with tables_col:
    if st.session_state.props_map_table is None:
        st.info("Press *Extract radial profiles* to compute the map.")
    else:
        df = st.session_state.props_map_table
        col_ctl, col_plot = st.columns([1, 2])
        with col_ctl:
            colour_by = st.selectbox("Colour by", df.columns, index=2, key="col_by")
            size_by   = st.selectbox("Size by",   df.columns, index=3, key="size_by")
            zmin, zmax = st.slider(
                "Colormap range (z)", 15, 30, (20, 22), key="zrange"
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
                on_select=lambda e: None,          # placeholder – real handler defined below
                selection_mode="points",
                key="sel_beams_plot",
            )
            # store selected filament id
            if sel and sel.selection and sel.selection.get("points"):
                pt = sel.selection["points"][0]
                st.session_state.selected_filament_index = pt["customdata"]
            else:
                st.session_state.selected_filament_index = None

# ------------------- Inspection tabs ----------------------------
if st.session_state.fil_table is not None:
    tab_filt, tab_table = st.tabs(["Inspect filament", "Properties table"])
    with tab_table:
        st.dataframe(st.session_state.fil_table)

    with tab_filt:
        if st.session_state.selected_filament_index is None:
            st.info("Select a filament on the map (right side).")
        else:
            fid = st.session_state.selected_filament_index[0]

            # three‑column layout for details
            col_prop, col_map, col_beam = st.columns([1, 2, 3])

            # ---- properties table ---------------------------------
            row = st.session_state.fil_table.loc[fid]
            row.index.name = "Fil‑ID"
            col_prop.write(row)

            # ---- selected filament map (interactive) ---------------
            df_sel = st.session_state.props_map_table
            df_fil = df_sel[df_sel["filID"] == fid]

            if len(df_fil) > 3:
                fig_one = plot_onefil_props_plotly(
                    st.session_state.cd_map.data,
                    df_fil,
                    crop_map=True,
                    size_by="W_bg",
                )
                beam_plot = col_map.plotly_chart(
                    fig_one,
                    key="selected_beam_plot",
                    selection_mode="points",
                    on_select=lambda e: None,
                )
                # store selected beam index
                if beam_plot and beam_plot.selection and beam_plot.selection.get("points"):
                    st.session_state.selected_beam_index = beam_plot.selection["points"][0]["point_index"]
                else:
                    st.session_state.selected_beam_index = None
            else:
                col_map.error("Not enough beams for a Plummer fit.")
else:
    st.info("Run Identification → Skeleton → Radial‑profile extraction first.")

# ------------------- Beam‑profile visualisation -----------------
if (
    st.session_state.selected_filament_index is not None
    and st.session_state.selected_beam_index is not None
    and st.session_state.local_field is not None
):
    fid = st.session_state.selected_filament_index[0]
    bid = st.session_state.selected_beam_index
    st.subheader(f"🔎 Filament {fid} – beam {bid}")

    fig = plot_radial_profile(
        st.session_state.local_field.filament_collection.filaments[fid],
        pl_indx=bid,
        getfig=True,
    )
    st.pyplot(fig)