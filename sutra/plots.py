from matplotlib import pyplot as plt 
from astropy.wcs import WCS 
from matplotlib.colors import LogNorm
from astropy import units as u
import numpy as np
import streamlit as st
import matplotlib.colors as clr

import plotly.express as px
import plotly.graph_objects as go 

# plt.style.use("dark")


# plt.

plt.style.use('dark_background')
plt.rcParams.update({"font.size":8})
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'


def plot_wcs(pacs_b_img , fig = None , figloc=111, norm=LogNorm(vmin=10, vmax=1000) ):
    pacs_b_img.data[pacs_b_img.data==0] = np.nan
    if fig == None:
        fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(figloc, projection = WCS(pacs_b_img.header))
    im = ax.imshow(pacs_b_img.data, cmap='inferno', norm=norm)

    ax.set_xlabel("Right Ascension", fontsize=12)
    ax.set_ylabel("Declination", fontsize=12)
    # ax.grid(color="white", ls='dotted', lw=2)

    # cbar = plt.colorbar(im, pad=0.1)
    # cbar.set_label(f"{pacs_b_img.header['BUNIT']}", size=16)

    # overlay = ax.get_coords_overlay('galactic')
    # overlay.grid(color='black', ls='dotted', lw=1)
    # overlay[0].set_axislabel('Galactic Longitude', fontsize=14)
    # overlay[1].set_axislabel('Galactic Latitude', fontsize=14)
    return fig, ax



def make_wcs_good(ax):
    axlist = [ax]
    for a in axlist:
        lon = a.coords[0]
        lat = a.coords[1]
        lon.set_axislabel(' ')
        lat.set_axislabel(' ')
        lon.display_minor_ticks(True)
        lat.display_minor_ticks(True)
        lon.set_major_formatter("d.dd")
        lat.set_major_formatter("d.dd")
        lon.set_ticklabel(exclude_overlapping=True)
    ax.set_xlabel("RA (J2000)",)
    ax.set_ylabel("Dec (J2000)", )
    return ax

# import matplotlib.colors as mcolors
import matplotlib.colors as mcolors
def bin_cmap(clr):
  """Creates a binary color palette for matplotlib where 0s are transparent 
  and 1s are red.

  Returns:
    A matplotlib ListedColormap object representing the binary palette.
  """

#   colors = [(0, 0, 0, 0), clr]  # Red is (1, 0, 0)
  cmap_name = 'binary_red'
  cmap = mcolors.LinearSegmentedColormap.from_list('bin',[(0,(0,0,0,0)), (1, clr)])
  return cmap


def plot_skl(pacs_b_img , fig = None , figloc=111, norm=LogNorm(vmin=10, vmax=1000) , cbar_title = None, cmap = 'magma'):
    if fig == None:
        fig = plt.figure(figsize=(6,6))
    wcs = WCS(pacs_b_img.header)
    # print(wcs)
    ax = fig.add_subplot(figloc, projection=wcs)
    im = ax.imshow(pacs_b_img.data,cmap=cmap)

    ax.set_xlabel("RA(J2000)",)
    ax.set_ylabel("Dec(J2000)", )
    ax.grid(color="black", ls='dotted', lw=1)

    return fig, ax

from astropy.wcs import WCS
from skimage.morphology import skeletonize
from scipy.signal import convolve2d
from skimage.morphology import skeletonize , dilation , footprints ,  dilation , ellipse, skeletonize , remove_small_objects



@st.cache_data
def compute_skeleton_image(skeleton, _header = None, cd=None, footprint_size =  1):
    # all heavy NumPy / SciPy work here, return plain arrays
    dilated = dilation(skeleton, ellipse(footprint_size, footprint_size))
    return dilated, cd

@st.cache_data
@st.cache_resource
def plot_skeleton(skeleton , _header = None , ax = None , cd = None):
	print("Plotting the skeleton again")
	dilated, cd_arr = compute_skeleton_image(skeleton, _header, cd)

	fig = plt.figure(figsize=(3.8, 3.8))
	ax = fig.add_subplot(111, projection=WCS(_header))

	if cd_arr is not None:
		ax.imshow(np.log10(cd_arr), cmap='Grays_r')
	ax.imshow(dilated, cmap=clr.LinearSegmentedColormap.from_list('bin',
					[(0,(0,0,0,0)), (1,(1,1,0,1))]))
	ax = make_wcs_good(ax)
	return fig

@st.cache_data
@st.cache_resource
def plot_props_map(data, df, size_by = "LMD", color_by = "FI",  _header = None):
	fig, ax = plt.subplots(1,1)
	ax.imshow(np.log10(data), origin = 'lower', cmap = 'Grays_r')

	size = np.power(df[size_by], 1) 
	size = 20*size / np.max(size)

	c = df[color_by] 
	c = c / np.max(c)

	ax.scatter(df['y'], df['x'],  s = size , c = c, cmap = 'inferno')
	# fig = plt.gcf()
	return fig




from matplotlib import cm
def map_to_colormap(array, colormap_name='viridis'):
    """
    Maps values in a NumPy array to a Matplotlib colormap.

    Args:
        array: A NumPy array of values.
        colormap_name: The name of the Matplotlib colormap to use (e.g., 'viridis', 'plasma', 'jet', 'coolwarm').
                    Defaults to 'viridis'.

    Returns:
        A NumPy array of RGBA colors corresponding to the input values,
        or a ValueError if the colormap name is invalid.
    """
    try:
        cmap = plt.get_cmap(colormap_name)
    except ValueError:
        raise ValueError(f"Invalid colormap name: {colormap_name}")

    norm = plt.Normalize(vmin=array.min(), vmax=array.max())  # Normalize to 0-1 range
    rgba_colors = cmap(norm(array))  # Map values to colors

    return rgba_colors
# @st.cache_resource

def get_beam_circ(cenx, ceny , rad):
        theta = np.linspace(-np.pi, np.pi)
        x  = cenx+rad*np.cos(theta)
        y  = ceny+rad*np.sin(theta)
        return x,y

from shapely.geometry import Point
from shapely.ops      import unary_union

def shapely_circle(cx, cy, radius, resolution=16):
    """
    Return a Shapely Polygon that approximates a circle.
    `resolution` is the number of line segments per quadrant
    (so total points ≈ 4*resolution).
    """
    return Point(cx, cy).buffer(radius, resolution=resolution)

# @st.cache_resource
def plot_onefil_props(data, df, size_by = "LMD", color_by = "FI",  _header = None, crop_map = False):
    fig, ax = plt.subplots(1,1)
    ax.imshow(np.log10(data), origin = 'lower', cmap = 'Grays_r')
    # if crop_map:
    print('sdsdsd')
    df = df[~df['rad_pix'].isna()]
    size = df[size_by] 
    size = 500 *size / np.max(size)

    c = df[color_by] 
    c = c / np.max(c)
    # c = map_to_colormap(c, colormap_name = 'Reds')


    cenx, ceny = df['x'] , df['y']
    rad = df['rad_pix']
    cmp = map_to_colormap(df[color_by] , 'Reds')
    circles = []
    for cx,cy,r , cl in zip(cenx,ceny,rad, cmp):
        x,y = get_beam_circ(cy,cx,r/2)
        circles.append(shapely_circle(cx, cy, r, resolution = 32))

    union_geom = unary_union(circles)   # this is a (Multi)Polygon object
    exteriors = [union_geom.exterior]
  
    if crop_map:
        xmin, ymin = np.inf, np.inf
        xmax, ymax = -np.inf, -np.inf

        for exterior in exteriors:
            xs, ys = exterior.xy                     # xs → column (x), ys → row (y)
            ax.plot(ys, xs, color='k', linewidth=1,
                        label='Outer boundary')
            xmin = min(xmin, np.min(xs))
            ymin = min(ymin, np.min(ys))
            xmax = max(xmax, np.max(xs))
            ymax = max(ymax, np.max(ys))

        margin = -1                      # pixels – change to whatever you like
        ax.set_ylim(xmin - margin, xmax + margin)
        ax.set_xlim(ymin - margin, ymax + margin)

    ax.scatter(df['y'], df['x'],  s = size , c = cmp, alpha = 0.5, lw=1 , )
    # ax.scatter(df['y'], df['x'],  s = size , c = 'white',  fillcolor=None, alpha = 0.5, lw=1 , )
    fig = plt.gcf()
    return fig



from tqdm.notebook import tqdm

# @st.cache_resource
def plot_props_map_plotly(data, df, size_by = "LMD", color_by = "FI",_header = None):

    fig = px.imshow(np.log10(data), color_continuous_scale = "gray", origin = 'lower', binary_string=True, binary_compression_level=1)
    df.loc[df[size_by].isna() , size_by] = 0
    df.loc[df[size_by]<0 , size_by] = 0

    size = np.power(df[size_by], 1) 
    size = 20*size / np.max(size)

    c = df[color_by] 
    c = c / np.max(c)

    fig.add_trace(
        go.Scatter(
            x = df['y'] , y = df['x'] , mode = 'markers', 
            customdata = df['filID'] , 
            marker = dict(size =size, color = c , colorscale = 'Reds', showscale = True), 
            name = "Filament Beam Elements", showlegend = False
        )
    )
    fig.update_layout(
        title = "FIlament Properties Map", 
    )
    fig.update_xaxes(showticklabels = False)
    fig.update_yaxes(showticklabels = False)
  
    findx_arr = (df['filID'].value_counts() > 4).index.to_list()
    for f in findx_arr:
        # print(f)
        circle_traces = []   # black outlines
        fill_traces   = []   # coloured fills
        shapely_circles = [] # for the union / outer boundary
        df_curr = df[df['filID']==f]
        cenx = df_curr['x'].values          # row coordinate (first axis)
        ceny = df_curr['y'].values          # column coordinate (second axis)
        rad  = df_curr['rad_pix'].values/2  # original code uses rad/2

        theta = np.linspace(0, 2*np.pi, 200)

        for cx, cy, r in zip(cenx, ceny, rad,):
            # points of the circle (same orientation as the original function)
            x = cx + r * np.cos(theta)      # row positions
            y = cy + r * np.sin(theta)      # column positions
            shapely_circles.append(shapely_circle(cx, cy, r, resolution=32))

        # ------------------------------------------------------------------
        # 2.4  Geometric union → outer boundary
        # ------------------------------------------------------------------
        union_geom = unary_union(shapely_circles)

        if union_geom.geom_type == 'Polygon':
            exteriors = [union_geom.exterior]
        else:   # MultiPolygon
            exteriors = [poly.exterior for poly in union_geom.geoms]

        # Outer‑boundary trace (single legend entry)
        boundary_traces = []
        for exterior in exteriors:
            xs, ys = exterior.xy                # xs → column (y), ys → row (x)
            boundary_traces.append(
                go.Scatter(
                    x=np.asarray(ys),
                    y=np.asarray(xs),
                    mode='lines',
                    line=dict(color='yellow', width=1),
                    # name='Outer boundary',
                    hoverinfo='skip', showlegend = False
                )
            )
        # fig.add_trace(scatter_trace)
        for tr in boundary_traces:
            fig.add_trace(tr)

    return fig
	# fig.update_layout(coloraxis_showscale = False)



@st.cache_resource  
def plot_props_map_plotly_v2(data, df, size_by = "LMD", color_by = "FI",_header = None,  highlight_colour="#ffea00" , skeleton = None , zmin=19, zmax = 23):
    # print('sdfsddsd')
    log_data = np.log10(data)
    # log_data = log_data / np.max(log_data)
    # skeleton , _ = compute_skeleton_image(skeleton, _header = None)
    # log_data = skeleton
    fig = px.imshow(log_data, color_continuous_scale = "inferno", origin = 'lower', binary_string=False, zmin=zmin, zmax=zmax,)
    fig.update_traces(hovertemplate = None)
    df.loc[df['rad_pix'].isna(), 'rad_pix'] = 0
    df.loc[df[size_by].isna() , size_by] = 0
    df.loc[df[size_by]<0 , size_by] = 0

   
    union_geoms = {}
    for fil_id, df_curr in df.groupby('filID'):
        # ---- make a Shapely circle for each beam belonging to this filament
        circles = [shapely_circle(row.x, row.y, row.rad_pix,
                                  resolution=32)
                   for row in df_curr.itertuples()]

        # ---- geometric union (may be a Polygon or a MultiPolygon) -----
        union = unary_union(circles)
        union_geoms[fil_id] = union

        # ---- extract the exterior(s) that we will actually draw -----
        if union.geom_type == "Polygon":
            exteriors = [union.exterior]
        else:                               # MultiPolygon
            exteriors = [poly.exterior for poly in union.geoms]

        # ---- we will draw **each exterior** as a *single* filled trace.
        #      The trace carries the filament summary data in `customdata`.
        # ----------------------------------------------------------------
        # Compute the aggregate quantities you asked for:
        total_len = df_curr['Length'].sum()
        total_mass = df_curr['Mass'].sum()

        # Build a (row, col) coordinate list that Plotly can fill.
        # The outer boundary may consist of several disjoint exteriors,
        # but Plotly can only fill one closed loop per trace.
        # Therefore we *concatenate* all exteriors with a NaN “break”
        # between them – Plotly will treat the whole thing as one polygon.
        xs, ys = [], []          # xs = column (→ x), ys = row (→ y)
        for exterior in exteriors:
            # exterior.xy returns (x‑coords, y‑coords) where
            #   x = column index, y = row index.
            col_xy, row_xy = exterior.xy
            xs.extend(col_xy.tolist())
            xs.append(np.nan)          # break between separate rings
            ys.extend(row_xy.tolist())
            ys.append(np.nan)

        # Convert to Plotly coordinates (swap axes)
        poly_x = xs          # column → Plotly *x*
        poly_y = ys          # row    → Plotly *y*

        # Attach the three values we want to expose on hover/selection:
        #   [filID, total_len, total_mass]
        custom = np.array([[fil_id, total_len, total_mass]],
                          dtype=object)      # shape (1,3)

        # ----------------------------------------------------------------
        #   Hover template (displayed when the mouse is over the polygon)
        # ----------------------------------------------------------------
        hover_tpl = (
            "<b>Filament ID:</b> %{customdata[0]}<br>"
            "<b>Total length:</b> %{customdata[1]:.2f}<br>"
            "<b>Total mass:</b>   %{customdata[2]:.2f}<extra></extra>"
        )

        # ----------------------------------------------------------------
        #   The trace itself – note the *selected* / *unselected* styles
        # ----------------------------------------------------------------
        
        trace = go.Scatter(
            x=poly_y,                     # <- horizontal axis
            y=poly_x,                     # <- vertical axis
            mode="lines",                 # we only draw the outline
            fill="toself",                # close the polygon and fill it
            # --------------------------------------------------------------
            # 1️⃣  DEFAULT (non‑selected) style
            # --------------------------------------------------------------
            fillcolor="rgba(255,250,124,0.2)",                # faint yellow
            line=dict(color="rgba(100,250,100,0.01)", width=1),
      

            # --------------------------------------------------------------
            # 2️⃣  HOVER (kept exactly as you already had)
            # --------------------------------------------------------------
            hovertemplate=hover_tpl,
            hoverlabel=dict(
                bgcolor="white",
                font_size=13,
                font_color="black",
                font_family="Arial",
            ),
            customdata=custom,            # filID, length, mass …
            name=str(fil_id),
            legendgroup=f"fil{fil_id}",
            showlegend=False,

            # --------------------------------------------------------------
            # 3️⃣  STYLE when the trace is **selected**
            # --------------------------------------------------------------
            selected=dict(
                # the fill colour becomes fully opaque
                # fillcolor="rgba(255,250,124,1)",          # bright yellow
                # make the outline more visible
                marker = dict(
                    # mode = 'lines', 
                    # fill = 'toself',
                    # line=dict(color="rgba(100,250,100,1)", width=2),

                ) , 
            ),
            # --------------------------------------------------------------
            # 4️⃣  STYLE when the trace is **not selected**
            # --------------------------------------------------------------
            # unselected=dict(
            #     # fillcolor="rgba(255,250,124,0.2)",        # faint yellow (default)
            #     line=dict(color="rgba(100,250,100,0.01)", width=1),
            # ),

            # --------------------------------------------------------------
            # 5️⃣  Required for selection to work (initially nothing is selected)
            # --------------------------------------------------------------
            selectedpoints=[],            # <-- will be filled by Plotly on click/box‑select
            hoverinfo="skip",             # we use hovertemplate instead of the default tooltip
        )

        fig.add_trace(trace)
    

    size = np.power(df[size_by], 1) 
    size = df[size_by]
    size = 20 * (size / np.max(size))

    c = df[color_by] 
    c = c / np.max(c)
    hover_tpl = (
        "<b>Filament ID:</b> %{customdata[0]}<br>"
        "<b>Total length:</b> %{customdata[1]:.2f}<br>"
        "<b>Total mass:</b>   %{customdata[2]:.2f}<extra></extra>"
        ),
    fig.add_trace(
        go.Scatter(
            x = df['y'] , y = df['x'] , 
           
            # # hoveron='fills', 
            # # hoverlabel=dict(bgcolor="white",
            # #                 font_size=13,
            # #                 font_family="Arial"),
            # marker = dict(size = size, color = c , colorscale = 'Reds', showscale = False, 
            #               line = dict(color = 'black', width =2) , 
            #               sizemode = 'diameter', 
            #             #   colorbar = False,
            #             # showscale = False,
            #               ), 
            unselected = dict(marker = dict(opacity=1)), 
            # customdata = df['filID'] , 
            # hovertemplate = f"<b>{size_by}</b>: %{{customdata[0]:.2f}}<br>" , 
            #         # f"<b>{color_by}</b>: %{{customdata[1]:.2f}}<extra></extra>",
            # # hoverinfo = 'skip',
            showlegend = False,


            mode='markers',
            # marker=dict(
            #     size=size_vec,
            #     color=fill_rgba,          # same RGBA used for the fills
            #     opacity=0.5,
            #     line=dict(width=1, color='black')
            # ),
            marker = dict(size = size, color = c , colorscale = 'Reds', showscale = False, 
                          line = dict(color = 'black', width =2) , 
                          sizemode = 'diameter', 
                        #   colorbar = False,
                        # showscale = False,
                          ), 
            # name='Filament points',
            hovertemplate= f"<b>Fil-ID</b>: %{{customdata[0]:.2f}}<br>",
            # f"<b>{color_by}</b>: %{{customdata[1]:.2f}}<extra></extra>",
            customdata = np.vstack(df['filID'].values,)

        )
    )
    fig.update_layout(coloraxis_showscale = False)

    
    fig.update_layout(
        title = "Filament Properties Map", 
    )
    fig.update_xaxes(showticklabels = False, visible = False)
    fig.update_yaxes(showticklabels = False, visible = False)
   

    fig.update_layout(coloraxis_showscale = False)
    fig.update_layout(
        xaxis=dict(visible=False, constrain='range'),   # hide tick labels
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1, constrain='range'),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        title='Filament with outer boundary (Plotly Express)'
    )
    
    # fig.update_layout(dragmode = 'zoom', clickmode = "event+select")
    
    # fig.update_layout(xaxis_fixedrange = True, yaxis_fixedrange = True ,margin=dict(l=0, r=0, t=0, b=0), )
    return fig








# ----------------------------------------------------------------------
# 2️⃣ Main routine – Plotly Express `imshow` + overlay traces
# ----------------------------------------------------------------------
@st.cache_resource
def plot_onefil_props_plotly(
        data, df,
        size_by="LMD",
        color_by="FI",
        crop_map=False,
        margin=10):
    """
    Parameters
    ----------
    data : 2‑D numpy array
        Column‑density map (displayed with log10 scaling).
    df   : pandas DataFrame
        Must contain at least the columns
        'x', 'y', 'rad_pix', plus the columns used for size/colour mapping.
    size_by, color_by : str
        Column names that control marker size and colour (same as the Matplotlib version).
    crop_map : bool
        If True, zoom the view to the exact outer boundary of the filament.
    margin : float
        Extra pixels added around the outer boundary when cropping.
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive Plotly figure (created with Plotly Express `imshow`).
    """
    # ------------------------------------------------------------------
    # 2.1  Log‑scale image with Plotly Express `imshow`
    # ------------------------------------------------------------------
    
    # ------------------------------------------------------------------
    # 2.2  Prepare colour & size vectors (exactly as before)
    # ------------------------------------------------------------------
    size_vec = np.power(df[size_by].values, 1)
    size_vec = 20 * size_vec / np.max(size_vec)

    # Normalise the colour column and map through Matplotlib's Reds colormap
    c_raw   = df[color_by].values.astype(float)
    c_norm  = c_raw / np.max(c_raw)
    cmap_r  = cm.get_cmap('Reds')
    fill_rgba = cmap_r(c_norm)                 # (N,4) in [0,1] range
    fill_hex = [f'rgba({int(r*255)},{int(g*255)},{int(b*255)},{a})'
                for r, g, b, a in fill_rgba]

    # ------------------------------------------------------------------
    # 2.3  Build circle traces (outline + filled polygon) and Shapely list
    # ------------------------------------------------------------------
    circle_traces = []   # black outlines
    fill_traces   = []   # coloured fills
    shapely_circles = [] # for the union / outer boundary

    cenx = df['x'].values          # row coordinate (first axis)
    ceny = df['y'].values          # column coordinate (second axis)
    rad  = df['rad_pix'].values / 2.0   # original code uses rad/2

    theta = np.linspace(0, 2*np.pi, 200)

    for cx, cy, r, col in zip(cenx, ceny, rad, fill_hex):
        # points of the circle (same orientation as the original function)
        x = cx + r * np.cos(theta)      # row positions
        y = cy + r * np.sin(theta)      # column positions
        shapely_circles.append(shapely_circle(cx, cy, r, resolution=32))

    # ------------------------------------------------------------------
    # 2.4  Geometric union → outer boundary
    # ------------------------------------------------------------------
    union_geom = unary_union(shapely_circles)

    if union_geom.geom_type == 'Polygon':
        exteriors = [union_geom.exterior]
    else:   # MultiPolygon
        exteriors = [poly.exterior for poly in union_geom.geoms]

    # Outer‑boundary trace (single legend entry)
    boundary_traces = []
    for exterior in exteriors:
        xs, ys = exterior.xy                # xs → column (y), ys → row (x)
        boundary_traces.append(
            go.Scatter(
                x=np.asarray(ys),
                y=np.asarray(xs),
                mode='lines',
                line=dict(color='lime', width=2),
                name='Outer boundary',
                hoverinfo='skip',
                showlegend=False
            )
        )


    log_data = np.log10(data)

    # Plotly‑Express automatically treats the array as an image trace.
    # We set `origin='lower'` so that the y‑axis points upward (like Matplotlib).

    if crop_map:
        # ---- compute the overall bounding box of the hull ----
        xmin, ymin = np.inf, np.inf   # rows (y‑axis) , cols (x‑axis)
        xmax, ymax = -np.inf, -np.inf
        for exterior in exteriors:
            xs, ys = exterior.xy          # xs → column, ys → row
            xmin = min(xmin, np.min(ys))  # row min → y‑axis range
            xmax = max(xmax, np.max(ys))  # row max → y‑axis range
            ymin = min(ymin, np.min(xs))  # col min → x‑axis range
            ymax = max(ymax, np.max(xs))  # col max → x‑axis range

        # ---- apply a small margin (same behaviour as the Matplotlib version) ----
        x_range = [ymin - margin, ymax + margin]   # Plotly *x* = column index
        y_range = [xmin - margin, xmax + margin]   # Plotly *y* = row index
    
    else: 
        xmin , xmax = 0 , log_data.shape[0]
        ymin , ymax = 0 , log_data.shape[1]
        margin = 0

    # ---- force the ranges (turn off autorange) ----
   
    fig = px.imshow(
        # log_data[int(ymin - margin):int(ymax - margin), int(xmin - margin):int(xmax - margin)],
        log_data,
        origin='lower',
        color_continuous_scale='twilight',
        zmin=float(log_data.min()),    # cast to native Python float
        zmax=float(log_data.max()),
        labels={'color': 'log10(data)'}, 
        binary_string=False, 
    )
    fig.update_traces(hovertemplate = None, hoverinfo  = 'skip')
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(autorange=False, range=y_range)
    fig.update_yaxes(autorange=False, range=x_range)



    # ------------------------------------------------------------------
    # 2.5  (Optional) scatter that mimics the commented‑out Matplotlib scatter
    # ------------------------------------------------------------------

    line_trace = go.Scatter(
        x=ceny,
        y=cenx,
        mode='lines',
        fillcolor = 'white',
        marker=dict(
            line=dict(width=10, color='white')
        ),
        showlegend=False ,
        name='Filament points',
        
    )
    fig.add_trace(line_trace)

    scatter_trace = go.Scatter(
        x=ceny,
        y=cenx,
        mode='markers',
        marker=dict(
            size=size_vec,
            color ='white' ,          # same RGBA used for the fills
            opacity=0.5,
            line=dict(width=1, color='white')
        ),
        showlegend=False ,
        name='Filament points',
        hovertemplate= f"<b>{size_by}</b>: %{{customdata[0]:.2f}}<br>" +
        f"<b>{color_by}</b>: %{{customdata[1]:.2f}}<extra></extra>",
        customdata=np.stack([df[size_by].values, df[color_by].values], axis=-1)
    )

    
    # ------------------------------------------------------------------
    # 2.6  Add all traces to the Plotly‑Express figure
    # ------------------------------------------------------------------
    # Plotly Express returns a Figure that already contains the image trace.
    # We simply append the extra Scatter traces.
    for tr in fill_traces + circle_traces:
        fig.add_trace(tr)

    fig.add_trace(scatter_trace)
    for tr in boundary_traces:
        fig.add_trace(tr)

    # ------------------------------------------------------------------
    # 2.7  Layout tweaks (hide axes, keep square aspect)
    # ------------------------------------------------------------------
    fig.update_layout(
        xaxis=dict(visible=False, constrain='range'),   # hide tick labels
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1, constrain='range'),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        title='Filament with outer boundary (Plotly Express)'
    )

    # ------------------------------------------------------------------
    # 2.8  Cropping to the exact outer boundary (if requested)
    # ------------------------------------------------------------------
     # --------------------------------------------------------------
    # 2.8  Cropping – now it really works!
    # --------------------------------------------------------------
   

    return fig




    