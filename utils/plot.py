"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 02.25

Description:
    Functions needed for plotting.
"""

from scipy.stats import binned_statistic_2d
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib import colors
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm
from pathlib import Path
import cycler
import plotly.graph_objects as go
import numpy as np
import plotly.express as px


def configure_matplotlib():
    """Configures Matplotlib with default settings and a specific font."""
    
    # Reset the plot configurations to default
    plt.rcdefaults()

    # Load a specific font
    font_path = str(Path(matplotlib.get_data_path(), "fonts/ttf/cmr10.ttf"))
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)

    # Apply font settings
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = prop.get_name()
    plt.rcParams["axes.formatter.use_mathtext"] = True
    plt.rcParams.update({'mathtext.default': 'regular'})

def configure_matplotlib_fabio(theme="light", figsize=(8,6)):
    """Configures Matplotlib with a publication-friendly style.
    
    Parameters:
    - theme (str): 'light' for white background, 'dark' for black background.
    """
    plt.rcdefaults()


    # Define custom color palette 
    custom_colors = ["#00A6FB", "#A559AA", "#14A76C", "#4634B2", "#FF4D80"]  
    plt.rcParams['axes.prop_cycle'] = cycler.cycler(color=custom_colors)

    # Basic font and figure settings
    plt.rcParams.update({
        'axes.titlesize': 18,
        'axes.labelsize': 20,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'lines.linewidth': 2,
        'lines.markersize': 6,
        'figure.figsize': [figsize[0], figsize[1]],
        'savefig.dpi': 300,
        'figure.dpi': 100,
        'figure.autolayout': True,
        'mathtext.default': 'regular',  # Ensure math expressions render correctly
    })

    # Theme-specific settings
    if theme == "dark":
        # Dark theme with light text for contrast
        plt.rcParams.update({
            'figure.facecolor': 'black', # Set the figure background to black
            'axes.facecolor': 'black',   # Background of the plot area
            'axes.edgecolor': 'white',   # Color of the axis lines
            'axes.labelcolor': 'white',  # Axis label color
            'xtick.color': 'white',      # Tick marks color
            'ytick.color': 'white',      # Tick marks color
            'text.color': 'white',       # General text color
            'grid.color': 'gray',        # Grid line color
            'grid.linestyle': '--',      # Grid line style
            'legend.facecolor': 'black', # Legend background
            'legend.edgecolor': 'white', # Legend border
            'legend.fontsize': 14,       # Legend font size
            'axes.unicode_minus': False, # Fix minus signs rendering
        })
    else:
        # Light theme with dark text for better readability
        plt.rcParams.update({
            'figure.facecolor': 'white', # Set the figure background to white
            'axes.facecolor': 'white',   # Background of the plot area
            'axes.edgecolor': 'black',   # Color of the axis lines
            'axes.labelcolor': 'black',  # Axis label color
            'xtick.color': 'black',      # Tick marks color
            'ytick.color': 'black',      # Tick marks color
            'text.color': 'black',       # General text color
            'grid.color': 'gray',        # Grid line color
            'grid.linestyle': '--',      # Grid line style
            'legend.facecolor': 'white', # Legend background
            'legend.edgecolor': 'black', # Legend border
            'legend.fontsize': 14,       # Legend font size
            'axes.unicode_minus': False, # Fix minus signs rendering
        })

    # Configure plot margins, padding, and layout
    plt.rcParams['axes.grid'] = False        # Enable gridlines
    plt.rcParams['figure.titlesize'] = 22   # Figure title size
    plt.rcParams['figure.titleweight'] = 'bold'  # Bold figure title for better emphasis


def plot_hits_3D(x, y, z, q, pred_lep = [], pred_seg = [], q_mode='categorical', primary_vertex=None, lepton_direction=None,
                 pdg=None, energy=None, ghost=False, s=1.8, plot_label=False, name_save_html=None):
    """
    Plots hits with an interactive 3D plot using Plotly.
    - 'categorical': Uses a colormap for different types of hits.
    - 'binary': Differentiates primary leptons from the rest.
    """
    fig = go.Figure()

    # CATEGORICAL MODE (0 = Ghost, 1 = EM, 2 = Hadronic)
    if q_mode == 'categorical':
        q = np.argmax(q, axis=1)
        color_map = {0: 'gray', 1: 'blue', 2: 'red'}
        size_map = {0: s * 1.2, 1: s * 1.4, 2: s * 1.8}  # Adjust size: FOR NOW THE SAME
        colors = [color_map[val] if val in color_map else 'black' for val in q]
        sizes = [size_map[val] if val in size_map else s for val in q]

        mask_ghost = q == 0
        mask_em = q == 1
        mask_had = q == 2

        # Plot 'EM' (Electromagnetic) category if there are any points
        if mask_em.sum() > 0:
            fig.add_trace(go.Scatter3d(
                x=z[mask_em], y=x[mask_em], z=y[mask_em],
                mode='markers',
                marker=dict(size=(s * 1), color='red', opacity=0.8),
                name='EM'
            ))

        # Plot 'GHOSt' category if there are any points
        if mask_ghost.sum() > 0:
            fig.add_trace(go.Scatter3d(
                x=z[mask_ghost], y=x[mask_ghost], z=y[mask_ghost],
                mode='markers',
                marker=dict(size=(s * 1), color='white', opacity=0.8),
                name='Ghost'
            ))

        # Plot 'HAD' (Hadronic) category if there are any points
        if mask_had.sum() > 0:
            fig.add_trace(go.Scatter3d(
                x=z[mask_had], y=x[mask_had], z=y[mask_had],
                mode='markers',
                marker=dict(size=(s * 1), color='blue', opacity=0.8),
                name='HAD'
            ))

        # Plot predictions as well
        if len(pred_seg) != 0:
            mask_pred_seg = np.argmax(pred_seg, axis=1)
            color_map = {0: 'black', 1: 'yellow', 2: 'green'}
            size_map = {0: s * 1.2, 1: s * 1.4, 2: s * 1.8}  # Adjust size: FOR NOW THE SAME
            colors = [color_map[val] if val in color_map else 'black' for val in q]
            sizes = [size_map[val] if val in size_map else s for val in q]

            mask_pred_ghost = mask_pred_seg == 0
            mask_pred_em = mask_pred_seg == 1
            mask_pred_had = mask_pred_seg == 2

            # Plot 'EM' (Electromagnetic) category if there are any points
            if mask_em.sum() > 0:
                fig.add_trace(go.Scatter3d(
                    x=z[mask_pred_em], y=x[mask_pred_em], z=y[mask_pred_em],
                    mode='markers',
                    marker=dict(size=(s * 1), color='green', opacity=0.8),
                    name='Pred EM'
                ))

            # Plot 'GHOSt' category if there are any points
            if mask_ghost.sum() > 0:
                fig.add_trace(go.Scatter3d(
                    x=z[mask_pred_ghost], y=x[mask_pred_ghost], z=y[mask_pred_ghost],
                    mode='markers',
                    marker=dict(size=(s * 1), color='gray', opacity=0.3),
                    name='Pred Ghost'
                ))

            # Plot 'HAD' (Hadronic) category if there are any points
            if mask_had.sum() > 0:
                fig.add_trace(go.Scatter3d(
                    x=z[mask_pred_had], y=x[mask_pred_had], z=y[mask_pred_had],
                    mode='markers',
                    marker=dict(size=(s * 1), color='yellow', opacity=0.8),
                    name='Pred HAD'
                ))
                    



    # BINARY MODE (Primary lepton vs Rest)
    elif q_mode == 'binary':
        mask_lepton = q[:, 0] == 1
        mask_rest = q[:, 0] == 0

        if mask_lepton.sum() > 0:
            fig.add_trace(go.Scatter3d(
                x=z[mask_lepton], y=x[mask_lepton], z=y[mask_lepton],
                mode='markers',
                marker=dict(size=(s + 0.5), color='orange', opacity=1.0),
                name='Primary Lepton'
            ))

        if mask_rest.sum() > 0:
            fig.add_trace(go.Scatter3d(
                x=z[mask_rest], y=x[mask_rest], z=y[mask_rest],
                mode='markers',
                marker=dict(size=s, color='white', opacity=0.3),
                name='Rest'
            ))

        # Plot predicitons as well
        if len(pred_lep) != 0:
            print("-------", len(pred_lep))
            mask_pred = pred_lep > 0.5
            mask_pred_lepton = mask_pred == 1
            mask_pred_rest = mask_pred == 0
            mask_pred_lepton = mask_pred_lepton.flatten()
            mask_pred_rest = mask_pred_rest.flatten()


            fig.add_trace(go.Scatter3d(
                x=z[mask_pred_lepton], y=x[mask_pred_lepton], z=y[mask_pred_lepton],
                mode='markers',
                marker=dict(size=(s + 0.5), color='aquamarine', opacity=0.8),
                name='Pred Lep'
            ))

            fig.add_trace(go.Scatter3d(
                x=z[mask_pred_rest], y=x[mask_pred_rest], z=y[mask_pred_rest],
                mode='markers',
                marker=dict(size=s, color='gray', opacity=0.3),
                name='Pred Rest'
            ))


    # ENERGY MODE (Color by sum of second and third column of q)
    elif q_mode == 'energy':
        energy_vals = q  # Sum second and third column
        min_energy, max_energy = np.min(energy_vals), np.max(energy_vals)

        # Get colors from the Viridis colormap
        color_scale = px.colors.sequential.Plotly3

        fig.add_trace(go.Scatter3d(
                x=z, y=x, z=y,
                mode='markers',
                marker=dict(
                    size=s, 
                    color=np.log(energy_vals.flatten()),  # Use normalized energy values for color mapping
                    colorscale=color_scale,
                    colorbar=dict(
                        title='Energy',  # Title for the color scale
                        ticktext=[f'{min_energy:.2f}', f'{max_energy:.2f}']  # Display min and max energy
                    ),
                    opacity=0.4,
                ),
                name='Energy-Based Coloring'
            ))

    # PRIMARY VERTEX
    if primary_vertex is not None and primary_vertex.shape == (3,):
        fig.add_trace(go.Scatter3d(
            x=[primary_vertex[2]], y=[primary_vertex[0]], z=[primary_vertex[1]],
            mode='markers',
            marker=dict(size=8, color='green', symbol='x'),
            name='Primary Vertex'
        ))

        # LEPTON DIRECTION (if provided)
        if lepton_direction is not None and lepton_direction.shape == (3,):
            end_point = primary_vertex + 2 * lepton_direction
            fig.add_trace(go.Scatter3d(
                x=[primary_vertex[2], end_point[2]],
                y=[primary_vertex[0], end_point[0]],
                z=[primary_vertex[1], end_point[1]],
                mode='lines',
                line=dict(color='green', width=4),
                name='Lepton Direction'
            ))

    # PLOT LABELS (PDG & Energy)
    if plot_label and pdg is not None and energy is not None:
        textstr_abs = f'Flavour: {pdg}\nEnergy: {energy:.2f} GeV'
        fig.add_annotation(
            text=textstr_abs,
            xref="paper", yref="paper",
            x=0.95, y=0.05,
            showarrow=False,
            font=dict(size=14),
            bgcolor="white",
            opacity=0.8
        )

    # 3D PLOT SETTINGS
    fig.update_layout(
        template='plotly_dark',
        scene=dict(
            xaxis_title='Z',
            yaxis_title='X',
            zaxis_title='Y',
            # xaxis=dict(range=[-1528, 1523]),
            # yaxis=dict(range=[-235, 235]),
            # zaxis=dict(range=[-235, 235]),
            aspectratio=dict(x=100, y=20, z=20)  # KEEPING aspect ratio
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        title='3D Hit Visualization',
        scene_camera=dict(
            eye=dict(x=2000, y=-2000, z=1200)  # KEEPING eye position
        ),
        showlegend=True,  # Ensure the legend is displayed
    )

    if name_save_html is not None:
        fig.write_html(name_save_html)
    fig.show()

def plot_hits(x, y, z, q, q_mode='categorical', primary_vertex=None, lepton_direction=None,
              pdg=None, energy=None, ghost=False, s=0.1, plot_label=False):
    """
    Plots hits with two different modes for `q` values:
    - 'categorical': Uses a colormap for different types of hits.
    - 'binary': Differentiates primary leptons from the rest.
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)

    if q_mode == 'categorical':
        cmap = ListedColormap(['gray', 'blue', 'red'])
        bounds = [-0.5, 0.5, 1.5, 2.5]  # Values just below and above the integers
        norm = BoundaryNorm(bounds, cmap.N)
        
        if ghost:
            ax.scatter(z, x, y, s=s, c="black", marker='o', alpha=0.7, label='Hits', zorder=2)
        else:
            sc = ax.scatter(z, x, y, s=s, c=q, cmap=cmap, norm=norm, marker='o', alpha=0.7, zorder=2)
        
        legend_elements = [
            Patch(facecolor='gray', label='Ghost'),
            Patch(facecolor='blue', label='Electromagnetic'),
            Patch(facecolor='red', label='Hadronic')
        ]
    
    elif q_mode == 'binary':
        mask_lepton = q[:, 0] == 1
        mask_rest = q[:, 0] == 0

        if mask_lepton.sum() > 0:
            ax.scatter(z[mask_lepton], x[mask_lepton], y[mask_lepton], s=s, c="orange", marker='o', alpha=1.0, label='Primary lepton', zorder=3)
        if mask_rest.sum() > 0:
            ax.scatter(z[mask_rest], x[mask_rest], y[mask_rest], s=s, c="black", marker='o', alpha=0.1, label='Rest', zorder=2)

        legend_elements = [
            Patch(facecolor='orange', label='Primary lepton'),
            Patch(facecolor='black', label='Rest')
        ]

    if primary_vertex is not None and primary_vertex.shape == (3,):
        ax.scatter(primary_vertex[2], primary_vertex[0], primary_vertex[1], s=200, c='green', marker='x', label='Primary Vertex', zorder=3)

        # Plot lepton direction if provided
        if lepton_direction is not None and lepton_direction.shape == (3,):
            end_point = primary_vertex + 2 * lepton_direction
            ax.quiver(primary_vertex[2], primary_vertex[0], primary_vertex[1],
                      end_point[2] - primary_vertex[2],
                      end_point[0] - primary_vertex[0],
                      end_point[1] - primary_vertex[1],
                      color='green', length=100, arrow_length_ratio=0.5, linewidth=2, zorder=5) 
    if plot_label:
        if primary_vertex is not None and primary_vertex.shape == (3,):
            legend_elements.append(plt.Line2D([0], [0], marker='x', color='green', linestyle='None', markersize=15, label='Primary Vertex'))
        if lepton_direction is not None and lepton_direction.shape == (3,):
            legend_elements.append(plt.Line2D([0], [0], marker='$\u2192$', color='green', linestyle='None', markersize=15, label='Primary lepton direction'))
        ax.legend(handles=legend_elements, loc='lower right', fontsize=20)
        if pdg is not None and energy is not None:
            textstr_abs = f'Flavour: {pdg}\nEnergy: {energy:.2f} GeV'
            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
            ax.text(5000, 10000, 1000, textstr_abs, transform=ax.transAxes,
                    fontsize=20, verticalalignment='top', horizontalalignment='right', bbox=props)
    
    ax.set_xlim(-1.5280e+03, 1.5236e+03)
    ax.set_ylim(-235, 235)
    ax.set_zlim(-235, 235)
    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    ax.set_box_aspect([300, 48, 48])
    plt.show()
