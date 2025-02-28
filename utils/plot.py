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

