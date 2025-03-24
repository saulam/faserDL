import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from dataset import SparseFASERCALDataset
from utils import ini_argparse
from torch.utils.data import DataLoader, Dataset
import torch
from utils.plot import configure_matplotlib_fabio
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

# -------------------------------------------------------------------------
# Custom dataset class
# -------------------------------------------------------------------------
class CustomDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        data = np.load(self.base_dataset.data_files[idx], allow_pickle=True)

        mask_is_lepton = data['primlepton_labels'] == 1
        mask_is_lepton = mask_is_lepton.flatten()
        mask_seg_lab = np.argmax(data['seg_labels'], axis=1)

        # find the min/max of z component of reco hits for leptons and non-leptons
        min_z_l = np.min(data['reco_hits'][mask_is_lepton, 2]) if mask_is_lepton.any() else np.nan
        max_z_l = np.max(data['reco_hits'][mask_is_lepton, 2]) if mask_is_lepton.any() else np.nan
        filtered_hits = data['reco_hits'][~mask_is_lepton, 2]

        min_z_nl = np.min(filtered_hits) if filtered_hits.size > 0 else np.nan
        max_z_nl = np.max(filtered_hits) if filtered_hits.size > 0 else np.nan

        dist_trav_lep = np.abs(min_z_l - max_z_l)
        dist_trav_non_lep = np.abs(min_z_nl - max_z_nl)


        result = {
            # Scalars
            'run_number': data['run_number'],
            'event_id': data['event_id'],
            'is_cc': data['is_cc'],
            'in_neutrino_pdg': data['in_neutrino_pdg'],
            'primary_vertex': data['primary_vertex'],
            'in_neutrino_energy': data['in_neutrino_energy'],
            # 'out_neutrino_energy': data['out_neutrino_energy'],
            'dist_trav_lep': dist_trav_lep,
            'min_z_l': min_z_l,
            'max_z_l': max_z_l,
            'dist_trav_non_lep': dist_trav_non_lep,
            'e_vis':data['e_vis'],

            # Non Scalar - per components
            'jet_momentum': data['jet_momentum'],



            # Non Scalars - per voxel
            'coords_lep': data['reco_hits'][mask_is_lepton,:3],
            'reco_module_lep': data['reco_hits'][mask_is_lepton, 3],
            'energy_lepton_vox': data['reco_hits'][mask_is_lepton, 4],
            'energy_non_lepton_vox': data['reco_hits'][~mask_is_lepton, 4],

            'energy_GH_vox': data['reco_hits'][mask_seg_lab == 0, 4],
            'energy_EM_vox': data['reco_hits'][mask_seg_lab == 1, 4], 
            'energy_HAD_vox': data['reco_hits'][mask_seg_lab == 2, 4], 
        }
        return result

# -------------------------------------------------------------------------
# Main script
# -------------------------------------------------------------------------
torch.multiprocessing.set_sharing_strategy('file_system')

# Parse arguments
parser = ini_argparse()
args = parser.parse_args()
# args.dataset_path = "/scratch3/salonso/faser/events_v3.5" #spaceml4
args.dataset_path = "/scratch/salonso/sparse-nns/faser/events_v3.5" #dlnu
plot_folder = "/home/fcufino/faserDL/Plots"

# Set GPU settings
nb_gpus = len(args.gpus)
gpus = ', '.join(args.gpus) if nb_gpus > 1 else str(args.gpus[0])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

# Load dataset
base_dataset = SparseFASERCALDataset(args)
dataset = CustomDataset(base_dataset)

# Custom collate function
def custom_collate_fn(batch):
    return batch

# DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=64,  
    collate_fn=custom_collate_fn,
    persistent_workers=True,
)

print(f"- Dataset size: {len(dataset)} events total.")


# Initialize storage for energy histograms, each initialized to zeros
# -------------------------------------------------------------------------
#arrays for aall the variables
min_z_e = []
max_z_e = []
min_z_mu = []
max_z_mu = []
min_z_tau = []
max_z_tau = []

dist_trav_lep_e = []
dist_trav_lep_mu = []
dist_trav_lep_tau = []

dist_trav_non_lep = []

energy_lepton_vox_e = []
energy_non_lepton_vox_e = []
energy_lepton_vox_mu = []
energy_non_lepton_vox_mu = []
energy_lepton_vox_tau = []
energy_non_lepton_vox_tau = []


energy_EM_e = [] 
energy_GH_e = []
energy_HAD_e = []

energy_EM_mu = []
energy_GH_mu = []
energy_HAD_mu = []

energy_EM_tau = []
energy_GH_tau = []
energy_HAD_tau = []

all_primary_vertex_z = []
energy_lepton_vox_e_vertex_z = []
module_lepton_vox_e_vertex_z = []
coord_vertex_z = []

n_ev = 0
total_events = len(base_dataset)

# Loop over dataloader
for batch in dataloader:
    # if n_ev > 100000:
    #     break

    for ev in batch:
        n_ev += 1

        neutrino_pdg = np.abs(ev['in_neutrino_pdg'])

        if ev['is_cc'] == 1:
            if neutrino_pdg == 12:
                energy_EM_e.append(ev['energy_EM_vox'])
                energy_GH_e.append(ev['energy_GH_vox'])
                energy_HAD_e.append(ev['energy_HAD_vox'])
                min_z_e.append(ev['min_z_l'])
                max_z_e.append(ev['max_z_l'])
                dist_trav_lep_e.append(ev['dist_trav_lep'])
                energy_lepton_vox_e.append(ev['energy_lepton_vox'])
                energy_non_lepton_vox_e.append(ev['energy_non_lepton_vox'])

                if 0 < ev['primary_vertex'][2] < 300:
                    energy_lepton_vox_e_vertex_z.append(ev['energy_lepton_vox'])
                    module_lepton_vox_e_vertex_z.append(ev['reco_module_lep'])
                    coord_vertex_z.append(ev['coords_lep'])


            elif neutrino_pdg == 14:
                energy_EM_mu.append(ev['energy_EM_vox'])
                energy_GH_mu.append(ev['energy_GH_vox'])
                energy_HAD_mu.append(ev['energy_HAD_vox'])
                min_z_mu.append(ev['min_z_l'])
                max_z_mu.append(ev['max_z_l'])
                dist_trav_lep_mu.append(ev['dist_trav_lep'])
                energy_lepton_vox_mu.append(ev['energy_lepton_vox'])
                energy_non_lepton_vox_mu.append(ev['energy_non_lepton_vox'])
            elif neutrino_pdg == 16:
                energy_EM_tau.append(ev['energy_EM_vox'])
                energy_GH_tau.append(ev['energy_GH_vox'])
                energy_HAD_tau.append(ev['energy_HAD_vox'])
                min_z_tau.append(ev['min_z_l'])
                max_z_tau.append(ev['max_z_l'])
                dist_trav_lep_tau.append(ev['dist_trav_lep'])
                energy_lepton_vox_tau.append(ev['energy_lepton_vox'])
                energy_non_lepton_vox_tau.append(ev['energy_non_lepton_vox'])
        else:
            dist_trav_non_lep.append(ev['dist_trav_non_lep'])
        

        # Print progress in percentage
        if n_ev % 1000 == 0:
            print(f"- Progress: {n_ev}/{total_events} ({n_ev/total_events:.1%})")



## PREPARATION ------------------------------------------------
configure_matplotlib_fabio(theme='dark')

# Define neutrino types for plotting
neutrino_pdg_map = {12: 'nue', 14: 'numu', 16: 'nutau'}

# Define a function for parallel concatenation
def concat(arrays):
    return np.concatenate(arrays, axis=0)

# -------------------------------------------------------------




# -------------------------------------------------------------------------
# Plot histogram of energy per module for different modules, with z range [0,200)
# -------------------------------------------------------------------------

data_lists = [module_lepton_vox_e_vertex_z, energy_lepton_vox_e_vertex_z, coord_vertex_z]
with ThreadPoolExecutor() as executor:
    all_module_lepton_vox_e, all_energy_lepton_vox_e, all_coord_vertex_z = executor.map(concat, data_lists)


plt.figure(figsize=(8, 6))
for i in range(6,10):  # Assuming 3 modules
    mask = all_module_lepton_vox_e == i  # Boolean mask
    plt.hist(all_energy_lepton_vox_e[mask] ,bins=100, label=f"Module {i}", histtype='step', lw = 2)

plt.legend()
plt.savefig(f'{plot_folder}/vox_energy_module1.png')
plt.close()

plt.figure(figsize=(8, 6))
for i in range(10,14):  # Assuming 3 modules
    mask = all_module_lepton_vox_e == i  # Boolean mask
    plt.hist(all_energy_lepton_vox_e[mask] ,bins=100, label=f"Module {i}", histtype='step', lw = 2)

plt.legend()
plt.savefig(f'{plot_folder}/vox_energy_module2.png')
plt.close()

# -------------------------------------------------------------------------
# Plot 2D histogram grid for projection distribution: ONLY mod 7
# -------------------------------------------------------------------------

mask_a = all_module_lepton_vox_e == 7
all_energy_lepton_vox_e = all_energy_lepton_vox_e[mask_a]
all_coord_vertex_z = all_coord_vertex_z[mask_a]

x_coords = all_coord_vertex_z[:,0]
y_coords = all_coord_vertex_z[:,1]
z_coords = all_coord_vertex_z[:,2]

def project_2d(x, y, energy, bin_size=25):
    """ Create a 2D histogram grid for projection with custom bin size. """
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    
    x_bins = np.arange(x_min, x_max + bin_size, bin_size)
    y_bins = np.arange(y_min, y_max + bin_size, bin_size)

    grid_energy, _, _ = np.histogram2d(x, y, bins=[x_bins, y_bins], weights=energy)

    return x_bins, y_bins, grid_energy.T


# Compute projections
x_bins_xy, y_bins_xy, energy_proj_xy = project_2d(x_coords, y_coords, all_energy_lepton_vox_e)
x_bins_xz, z_bins_xz, energy_proj_xz = project_2d(x_coords, z_coords, all_energy_lepton_vox_e)
y_bins_yz, z_bins_yz, energy_proj_yz = project_2d(y_coords, z_coords, all_energy_lepton_vox_e)

# Plot projections with a grid
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# XY Projection
mesh1 = axes[0].pcolormesh(x_bins_xy, y_bins_xy, energy_proj_xy, cmap='inferno', shading='auto')
axes[0].set_xlabel("X")
axes[0].set_ylabel("Y")
axes[0].set_title("XY Projection (Summed over Z)")
fig.colorbar(mesh1, ax=axes[0], label="Energy Deposited")

# XZ Projection
mesh2 = axes[1].pcolormesh(x_bins_xz, z_bins_xz, energy_proj_xz, cmap='inferno', shading='auto')
axes[1].set_xlabel("X")
axes[1].set_ylabel("Z")
axes[1].set_title("XZ Projection (Summed over Y)")
fig.colorbar(mesh2, ax=axes[1], label="Energy Deposited")

# YZ Projection
mesh3 = axes[2].pcolormesh(y_bins_yz, z_bins_yz, energy_proj_yz, cmap='inferno', shading='auto')
axes[2].set_xlabel("Y")
axes[2].set_ylabel("Z")
axes[2].set_title("YZ Projection (Summed over X)")
fig.colorbar(mesh3, ax=axes[2], label="Energy Deposited")

plt.tight_layout()
plt.savefig(f'{plot_folder}/event_projections_grid.png')
plt.show()
plt.close()

# -------------------------------------------------------------------------
# Plot the MIN_Z distribution
# -------------------------------------------------------------------------
# nue
plt.figure(figsize=(8, 6))
plt.hist(min_z_mu, bins=100,  label='numu')
plt.hist(min_z_e, bins=100, label='nue')
plt.hist(min_z_tau, bins=100,  label='nutau')
plt.xlabel('$z_{min}$ [mm]')
plt.ylabel('Counts')
plt.yscale('log')
plt.legend(loc='upper left')
plt.savefig(f'{plot_folder}/min_z_distribution.png')
plt.close()


# -------------------------------------------------------------------------
# Plot the MAX_Z distribution
# -------------------------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.hist(max_z_mu, bins=100, label='numu')
plt.hist(max_z_e, bins=100,  label='nue')
plt.hist(max_z_tau, bins=100, label='nutau')
plt.xlabel('$z_{max}$ [mm]')
plt.ylabel('Counts')
plt.yscale('log')
plt.legend()
plt.savefig(f'{plot_folder}/max_z_distribution.png')
plt.close()


# -------------------------------------------------------------------------
# Plot the DIST_TRAV_LEP vs NON TRAV distribution
# -------------------------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.hist(dist_trav_lep_mu, bins=110, range=(0,2000), label='numu')
plt.hist(dist_trav_lep_e, bins=110, range=(0,2000),  label='nue')
plt.hist(dist_trav_lep_tau, bins=110, range=(0,2000), label='nutau')
#plot white line at max value of dist_trav_lep_mu orizzontal
plt.axhline(y=800, color='white', linestyle='--', linewidth=2)
plt.xlabel('$z_{dist} = |z_{max} - z_{min}|$ [mm]')
plt.ylabel('Counts')
plt.yscale('log')
plt.legend(loc = 'lower right')
plt.savefig(f'{plot_folder}/dist_trav_lep_distribution.png')
plt.close()


# -------------------------------------------------------------------------
# Plot Vox lep energy vs Vox non lep energy
# -------------------------------------------------------------------------

# Define datasets
data_groups = [
    (energy_lepton_vox_e, energy_non_lepton_vox_e),
    (energy_lepton_vox_mu, energy_non_lepton_vox_mu),
    (energy_lepton_vox_tau, energy_non_lepton_vox_tau),
]

# Define a function for concatenation
def concat(arrays):
    return np.concatenate(arrays, axis=0)

# Run all concatenations in parallel
with ThreadPoolExecutor() as executor:
    results = list(executor.map(concat, [pair for group in data_groups for pair in group]))

# Unpack results
(all_hist_energy_lepton_vox_e, all_hist_energy_non_lepton_vox_e,
 all_hist_energy_lepton_vox_mu, all_hist_energy_non_lepton_vox_mu,
 all_hist_energy_lepton_vox_tau, all_hist_energy_non_lepton_vox_tau) = results

# Print shapes
print(all_hist_energy_lepton_vox_e.shape, all_hist_energy_non_lepton_vox_e.shape,
      all_hist_energy_lepton_vox_mu.shape, all_hist_energy_non_lepton_vox_mu.shape,
      all_hist_energy_lepton_vox_tau.shape, all_hist_energy_non_lepton_vox_tau.shape)

# Define the number of bins
num_bins = 100

# Compute log-spaced bins based only on primlepton E values
min_val = np.min(all_hist_energy_lepton_vox_e)
max_val = np.max(all_hist_energy_lepton_vox_e)
bin_edges = np.logspace(np.log10(min_val), np.log10(max_val), num_bins + 1)

plt.figure(figsize=(8,6))
# Plot only the primlepton E histogram
plt.hist(all_hist_energy_lepton_vox_e, bins=bin_edges, label='primlepton E', color='purple')
plt.axvline(66, 0, 3500, color = 'white', label = '66 MeV, Low-Gain ch')
# Labels and scales
plt.xlabel('Vox Energy [MeV]')
plt.ylabel('Counts')
plt.xscale('log')
plt.legend()

# Save the plot
plt.savefig(f'{plot_folder}/vox_energy_distribution_primlepton.png')
plt.close()

num_bins = 100

# Compute log-spaced bins based only on non-lepton values
min_val = np.min(all_hist_energy_non_lepton_vox_e)
max_val = np.max(all_hist_energy_non_lepton_vox_e)
bin_edges = np.logspace(np.log10(min_val), np.log10(max_val), num_bins + 1)

plt.figure(figsize=(8,6))
# Plot only the non-lepton histogram
plt.hist(all_hist_energy_non_lepton_vox_e, bins=bin_edges, label='non lepton')
plt.axvline(66, 0, 2500000, color = 'white', label = '66 MeV, Low-Gain ch')
# Labels and scales
plt.xlabel('Vox Energy [MeV]')
plt.ylabel('Counts')
plt.xscale('log')
plt.legend()

# Save the plot
plt.savefig(f'{plot_folder}/vox_energy_distribution_non_lepton.png')
plt.close()

# nue
# Define the number of bins
num_bins = 100

# Plot histograms using predefined bins
plt.hist(all_hist_energy_non_lepton_vox_e,
          bins=np.logspace(np.log10(min(np.min(all_hist_energy_non_lepton_vox_e), np.min(all_hist_energy_lepton_vox_e))),
                           np.log10(max(np.max(all_hist_energy_non_lepton_vox_e), np.max(all_hist_energy_lepton_vox_e))),
                           num_bins + 1),
          label='non lepton',
          lw=2)
plt.hist(all_hist_energy_lepton_vox_e,
          bins=np.logspace(np.log10(min(np.min(all_hist_energy_non_lepton_vox_e), np.min(all_hist_energy_lepton_vox_e))),
                           np.log10(max(np.max(all_hist_energy_non_lepton_vox_e), np.max(all_hist_energy_lepton_vox_e))),
                           num_bins + 1),
          label='primlepton E',
          lw=2)
plt.xlabel('Vox Energy [MeV]')
plt.ylabel('Counts')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig(f'{plot_folder}/vox_energy_distribution_nue.png')
plt.close()

# numu
plt.hist(all_hist_energy_non_lepton_vox_mu,
          bins=np.logspace(np.log10(min(np.min(all_hist_energy_non_lepton_vox_mu), np.min(all_hist_energy_lepton_vox_mu))),
                           np.log10(max(np.max(all_hist_energy_non_lepton_vox_mu), np.max(all_hist_energy_lepton_vox_mu))),
                           num_bins + 1),
          label='non lepton')
plt.hist(all_hist_energy_lepton_vox_mu,
          bins=np.logspace(np.log10(min(np.min(all_hist_energy_non_lepton_vox_mu), np.min(all_hist_energy_lepton_vox_mu))),
                           np.log10(max(np.max(all_hist_energy_non_lepton_vox_mu), np.max(all_hist_energy_lepton_vox_mu))),
                           num_bins + 1),
          label='primlepton Mu')
plt.xlabel('Vox Energy [MeV]')
plt.ylabel('Counts')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig(f'{plot_folder}/vox_energy_distribution_numu.png')
plt.close()

# nutau
plt.hist(all_hist_energy_non_lepton_vox_tau,
          bins=np.logspace(np.log10(min(np.min(all_hist_energy_non_lepton_vox_tau), np.min(all_hist_energy_lepton_vox_tau))),
                           np.log10(max(np.max(all_hist_energy_non_lepton_vox_tau), np.max(all_hist_energy_lepton_vox_tau))),
                           num_bins + 1),
          label='non lepton')
plt.hist(all_hist_energy_lepton_vox_tau,
          bins=np.logspace(np.log10(min(np.min(all_hist_energy_non_lepton_vox_tau), np.min(all_hist_energy_lepton_vox_tau))),
                           np.log10(max(np.max(all_hist_energy_non_lepton_vox_tau), np.max(all_hist_energy_lepton_vox_tau))),
                           num_bins + 1),
          label='primlepton Tau')
plt.xlabel('Vox Energy [MeV]')
plt.ylabel('Counts')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig(f'{plot_folder}/vox_energy_distribution_nutau.png')
plt.close()



# -------------------------------------------------------------------------

# Define datasets in groups
data_groups = [
    (energy_EM_e, energy_GH_e, energy_HAD_e),
    (energy_EM_mu, energy_GH_mu, energy_HAD_mu),
    (energy_EM_tau, energy_GH_tau, energy_HAD_tau),
]

# Define a function for concatenation
def concat(arrays):
    return np.concatenate(arrays, axis=0)

# Run all concatenations in parallel
with ThreadPoolExecutor() as executor:
    results = list(executor.map(concat, [item for group in data_groups for item in group]))

# Unpack results
(all_hist_energy_EM_e, all_hist_energy_GH_e, all_hist_energy_HAD_e,
 all_hist_energy_EM_mu, all_hist_energy_GH_mu, all_hist_energy_HAD_mu,
 all_hist_energy_EM_tau, all_hist_energy_GH_tau, all_hist_energy_HAD_tau) = results

# Print shapes
print(all_hist_energy_EM_e.shape, all_hist_energy_GH_e.shape, all_hist_energy_HAD_e.shape,
      all_hist_energy_EM_mu.shape, all_hist_energy_GH_mu.shape, all_hist_energy_HAD_mu.shape,
      all_hist_energy_EM_tau.shape, all_hist_energy_GH_tau.shape, all_hist_energy_HAD_tau.shape)

# Create and save the plots for summed histograms
# nue
plt.figure()
plt.hist(all_hist_energy_EM_e, bins=100,  label='EM')
plt.hist(all_hist_energy_HAD_e, bins=100,  label='HAD')
plt.hist(all_hist_energy_GH_e, bins=100,  label='GH') 
plt.xlabel('Vox Energy [MeV]')
plt.ylabel('Counts')
plt.yscale('log')
plt.legend()
plt.savefig(f'{plot_folder}/energy_distributions_nue.png')
plt.close()

# numu
plt.hist(all_hist_energy_EM_mu, bins=100,   label='EM')
plt.hist(all_hist_energy_HAD_mu, bins=100,  label='HAD')
plt.hist(all_hist_energy_GH_mu, bins=100,  label='GH')
plt.xlabel('Vox Energy [MeV]')
plt.ylabel('Counts')
plt.yscale('log')
plt.legend()
plt.savefig(f'{plot_folder}/energy_distributions_numu.png')
plt.close()

# nutau
plt.figure()
plt.hist(all_hist_energy_EM_tau, bins=100,   label='EM')
plt.hist(all_hist_energy_HAD_tau, bins=100, label='HAD')
plt.hist(all_hist_energy_GH_tau, bins=100,   label='GH')
plt.xlabel('Vox Energy [MeV]')
plt.ylabel('Counts')
plt.yscale('log')
plt.legend()
plt.savefig(f'{plot_folder}/energy_distributions_nutau.png')
plt.close()
