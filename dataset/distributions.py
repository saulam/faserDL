import os
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
from torch.utils.data import DataLoader, Dataset
import torch
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataset import SparseFASERCALDataset
from utils import ini_argparse
from torch.nn.utils.rnn import pad_sequence
from utils.plot import configure_matplotlib_fabio
from utils.augmentations import *


# -------------------------------------------------------------------------
# Custom dataset class
# -------------------------------------------------------------------------
class CustomDataset(SparseFASERCALDataset):
    def __init__(self, args):
        super().__init__(args)  # This will call the __init__ method of SparseFASERCALDataset
        
        # Initialize other attributes specific to CustomDataset
        self.plot_distributions = True


    def __getitem__(self, idx):
        """
        Retrieves a data sample by index and applies augmentations if needed.
        """
        data = super().__getitem__(idx)
        

        #fixig some stuff
        data['primlepton_labels'] = data['feats'][:,1]
        data['seg_labels'] = data['feats'][:,2:5]

        data['rear_cal_energy'] = data['feats_global'][0]
        data['rear_hcal_energy'] = data['feats_global'][1]
        data['rear_mucal_energy'] = data['feats_global'][2]
        data['faser_cal_energy'] = data['feats_global'][3]

  
        mask_is_lepton = (data['primlepton_labels'] == 1).flatten()
        mask_seg_lab = np.argmax(data['seg_labels'], axis=1)

        def get_z_range(mask):
            if mask.any():
                z_vals = data['coords'][mask, 2]
                return torch.min(z_vals), torch.max(z_vals)
            return np.nan, np.nan

        min_z_l, max_z_l = get_z_range(mask_is_lepton)
        min_z_nl, max_z_nl = get_z_range(~mask_is_lepton)

        
        output =  {
            **{key: data[key] for key in [
                'run_number', 'event_id', 'primary_vertex', 'is_cc', 'in_neutrino_pdg', 
                'in_neutrino_energy', 'out_lepton_momentum_dir', 'flavour_label',
                'e_vis', 'pt_miss', 'out_lepton_momentum_mag', 'jet_momentum_mag', 'jet_momentum_dir', 'coords', 'feats', 'feats_global',
                'primlepton_labels', 'seg_labels',
                'rear_cal_energy', 'rear_hcal_energy', 'rear_mucal_energy', 'faser_cal_energy']},

            'energy_lepton_vox': data['feats'][mask_is_lepton, 0],
            'energy_non_lepton_vox': data['feats'][~mask_is_lepton, 0],
            'energy_GH_vox': data['feats'][mask_seg_lab == 0, 0],
            'energy_EM_vox': data['feats'][mask_seg_lab == 1, 0],
            'energy_HAD_vox': data['feats'][mask_seg_lab == 2, 0],
            'min_z_l': min_z_l,
            'max_z_l': max_z_l,
            'dist_trav_lep': abs(max_z_l - min_z_l),
            'dist_trav_non_lep': abs(max_z_nl - min_z_nl),
        }

        # Clean up data variables to free memory before returning the output
        del data
        del mask_is_lepton, mask_seg_lab, min_z_l, max_z_l, min_z_nl, max_z_nl

        return output



# -------------------------------------------------------------------------
# Setup and arguments
# -------------------------------------------------------------------------
torch.multiprocessing.set_sharing_strategy('file_system')
args = ini_argparse().parse_args()

# args.dataset_path = "/scratch3/salonso/faser/events_v3.5" #spaceml4
# args.dataset_path = "/scratch/salonso/sparse-nns/faser/events_v3.5" #dlnu
args.dataset_path = "/scratch2/salonso/faser/events_v5.1"

args.train = False
args.stage1 = False
args.augmentations_enabled = False
args.batch_size = 4
args.num_workers = 32

plot_folder = "/home/fcufino/faserDL/Plots/Plotsv5_1"

# plot_folder = "/home/fcufino/faserDL/Plots/Plotsv3_5_AUG"
# plot_folder = "/home/fcufino/faserDL/Plots/Plotsv3_5"


# GPU
gpus = ','.join(args.gpus) if len(args.gpus) > 1 else str(args.gpus[0])
os.environ.update({
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "CUDA_VISIBLE_DEVICES": gpus
})


# Check if CUDA is available and select device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Print out the device being used
if torch.cuda.is_available():
    print(f"Running on GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Running on CPU")



# -------------------------------------------------------------------------
# Loop over all the events: setup 
# -------------------------------------------------------------------------

def collate_test(batch):
    """
    Collate function for batching, handles tensor concatenation and other optional keys.
    """
    ret = {
        # 'f': torch.cat([d['feats'] for d in batch]),
        # 'f_glob': torch.stack([d['feats_global'] for d in batch]),
        # 'c': [d['coords'] for d in batch],  # Store coords as a list for later padding if needed
    }
    
    # Optional keys to include in the batch
    optional_keys = [
        # Info ev
        'run_number', 'event_id', 'primary_vertex', 'is_cc', 'in_neutrino_pdg', 
         
        # Labels
        'primlepton_labels', 'seg_labels', 'flavour_label',
        
        # Per event values
        'in_neutrino_energy', 
        'e_vis', 'pt_miss',
        'rear_cal_energy', 'rear_hcal_energy', 'rear_mucal_energy', 'faser_cal_energy',
        'out_lepton_momentum_dir', 'jet_momentum_dir',
        
        # Per voxel values
        'energy_lepton_vox','energy_non_lepton_vox','energy_GH_vox',
        'energy_EM_vox','energy_HAD_vox','min_z_l','max_z_l','dist_trav_lep','dist_trav_non_lep'
    ]

    # Loop through optional keys and add to the return dict
    for key in optional_keys:
        if key in batch[0]:  # Check if the key exists in the batch
            if key in ['primlepton_labels', 'seg_labels', 'flavour_label',
                       'energy_lepton_vox','energy_non_lepton_vox','energy_GH_vox',
                        'energy_EM_vox','energy_HAD_vox']:
                # If it's a label, store as numpy array
                ret[key] = [d[key].numpy() for d in batch]
            elif key in ['e_vis', 'pt_miss', 
                         'rear_cal_energy', 'rear_hcal_energy', 'rear_mucal_energy', 'faser_cal_energy']:
                # If it's a scalar value, use `.item()` to get the value
                ret[key] = [d[key].item() for d in batch]
            else:
                # For all other keys, store the tensor as is
                ret[key] = [d[key] for d in batch]

    return ret



# prefetch_factor=1 was important
dataset = CustomDataset(args)
dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                        pin_memory=True, persistent_workers=True if args.num_workers > 0 else False, prefetch_factor=1,
                        collate_fn=collate_test)
print(f"- Dataset size: {len(dataset)} events total.")



# -------------------------------------------------------------------------
# Storage setup using dictionary
# -------------------------------------------------------------------------
neutrino_data = {k: defaultdict(list) for k in ['e', 'mu', 'tau', 'NC', 'shared']}
neutrino_map = {12: 'e', 14: 'mu', 16: 'tau'}

n_ev = 0

for batch in dataloader:
    if n_ev > 5000:
        break

    len_batch = len(batch['run_number'])

    for i in range(len_batch):
        ev = {key: batch[key][i] for key in batch}
        
        n_ev += 1

        pdg = abs(ev['in_neutrino_pdg'])
        key = neutrino_map.get(pdg, 'NC') if ev['is_cc'] else 'NC'
        store = neutrino_data[key]

        for var in [
            'in_neutrino_energy', 'e_vis', 'pt_miss',
            'rear_cal_energy', 'rear_hcal_energy', 'rear_mucal_energy', 'faser_cal_energy',
            'primary_vertex', 'jet_momentum_dir',
            'out_lepton_momentum_dir']:
            store[var].append(ev[var])

        if key != 'NC':
            for var in [
                'energy_EM_vox', 'energy_GH_vox', 'energy_HAD_vox',
                'min_z_l', 'max_z_l', 'dist_trav_lep',
                'energy_lepton_vox', 'energy_non_lepton_vox']:
                store[var].append(ev[var])

        if not ev['is_cc']:
            neutrino_data['NC']['dist_trav_non_lep'].append(ev['dist_trav_non_lep'])

        if n_ev % 1000 == 0:
            print(f"- Progress: {n_ev}/{len(dataset)} ({n_ev/len(dataset):.1%})")

        torch.cuda.empty_cache()
    del batch


# -------------------------------------------------------------
configure_matplotlib_fabio(theme='dark')
# -------------------------------------------------------------





# -------------------------------------------------------------
# Plotting functions
# -------------------------------------------------------------
def plot_1d(variable_data, variable_name, plot_filename, xlabel='Energy [GeV]', log_scale=False, n=4):
    # Define colors for each dataset
    colors = ["#00A6FB", "#A559AA", "#14A76C", "#4634B2"]
    neutrino_types = ['NuE', 'NuMu', 'NuTau', 'NC']

    plt.figure(figsize=(4 * n, 4))

    # Ensure correct shape of data, flatten if necessary
    for i, data in enumerate(variable_data):
        # Convert data to numpy array if it's not already
        data = np.array(data)
        
        if len(data.shape) > 1:
            data = data.flatten()  # Flatten to 1D if it's multidimensional
        
        # Plot the data with the correct color
        plt.subplot(1, n, i + 1)
        plt.hist(data, bins=40, label=neutrino_types[i], color=colors[i])  # Provide the correct color
        plt.title(f"{neutrino_types[i]} {variable_name}")
        plt.xlabel(xlabel)
        plt.ylabel('Counts')
        if log_scale:
            plt.yscale('log')
        plt.legend()

    plt.tight_layout()
    plt.savefig(f'{plot_folder}/{plot_filename}.png')
    plt.close()


# -------------------------------------------------------------
# Call plotting functions for each variable
# -------------------------------------------------------------
# plot_1d([
#     neutrino_data['e']['in_neutrino_energy'], neutrino_data['mu']['in_neutrino_energy'],
#     neutrino_data['tau']['in_neutrino_energy'], neutrino_data['NC']['in_neutrino_energy']
# ], "Incoming Nu Energy", "in_neutrino_energy_1x4")


plot_1d([
    neutrino_data['e']['e_vis'], neutrino_data['mu']['e_vis'],
    neutrino_data['tau']['e_vis'], neutrino_data['NC']['e_vis']
], "E_vis", "e_vis_1x4")

plot_1d([
    neutrino_data['e']['pt_miss'], neutrino_data['mu']['pt_miss'],
    neutrino_data['tau']['pt_miss'], neutrino_data['NC']['pt_miss']
], "pt_miss", "pt_miss_1x4")

print('---- Done')

plot_1d([
    neutrino_data['e']['rear_cal_energy'], neutrino_data['mu']['rear_cal_energy'],
    neutrino_data['tau']['rear_cal_energy'], neutrino_data['NC']['rear_cal_energy']
], "Rear Cal Energy", "rear_cal_energy_1x4")

plot_1d([
    neutrino_data['e']['rear_hcal_energy'], neutrino_data['mu']['rear_hcal_energy'],
    neutrino_data['tau']['rear_hcal_energy'], neutrino_data['NC']['rear_hcal_energy']
], "Rear Hcal Energy", "rear_hcal_energy_1x4", log_scale=True)

plot_1d([
    neutrino_data['e']['rear_mucal_energy'], neutrino_data['mu']['rear_mucal_energy'],
    neutrino_data['tau']['rear_mucal_energy'], neutrino_data['NC']['rear_mucal_energy']
], "Rear Mucal Energy", "rear_mucal_energy_1x4", log_scale=True)

plot_1d([
    neutrino_data['e']['faser_cal_energy'], neutrino_data['mu']['faser_cal_energy'],
    neutrino_data['tau']['faser_cal_energy'], neutrino_data['NC']['faser_cal_energy']
], "Faser Cal Energy", "faser_cal_energy_1x4")

# -------------------------------------------------------------
# 3D plotting for spatial components (e.g., primary vertex, momentum)
# -------------------------------------------------------------
def plot_3d(variable_data, variable_name, plot_filename):
    # Colors for each component (x, y, z)
    colors = ["#00A6FB", "#A559AA", "#14A76C"]

    # Iterate over each neutrino type (e, mu, tau, NC)
    for i, data in enumerate(variable_data):
        plt.figure(figsize=(12, 4))  # Create a new figure for each neutrino type

        # Convert to numpy array if it's not already
        data = np.array(data)

        # Reshape 1D data if necessary
        if data.ndim == 1:
            data = data.reshape(-1, 3)  # Reshape to (n, 3) if it's 1D (flat array)

        # Extract the x, y, z components
        x_data = data[:, 0]
        y_data = data[:, 1]
        z_data = data[:, 2]

        # Create a subplot for each component (x, y, z)
        for j, component in enumerate([x_data, y_data, z_data]):
            plt.subplot(1, 3, j + 1)  # 1x3 grid, plot component j in the correct position
            plt.hist(component, bins=50, label=['X', 'Y', 'Z'][j], color=colors[j])
            plt.title(f"{variable_name} ({['NuE', 'NuMu', 'NuTau', 'NC'][i]})")
            plt.xlabel('Component Value')
            plt.ylabel('Counts')
            plt.legend()

        plt.tight_layout()
        plt.savefig(f'{plot_folder}/{plot_filename}_{["NuE", "NuMu", "NuTau", "NC"][i]}.png')
        plt.close()



# -------------------------------------------------------------
# Example 3D plot calls
# -------------------------------------------------------------
# Apply plot_3d function for all variables
plot_3d([
    neutrino_data['e']['primary_vertex'], 
    neutrino_data['mu']['primary_vertex'],
    neutrino_data['tau']['primary_vertex'], 
    neutrino_data['NC']['primary_vertex']
], "Primary Vertex", "primary_vertex_1x3")

plot_3d([
    neutrino_data['e']['jet_momentum_dir'], 
    neutrino_data['mu']['jet_momentum_dir'],
    neutrino_data['tau']['jet_momentum_dir'], 
    neutrino_data['NC']['jet_momentum_dir']
], "Jet Momentum", "jet_momentum_1x3")

plot_3d([
    neutrino_data['e']['out_lepton_momentum_dir'], 
    neutrino_data['mu']['out_lepton_momentum_dir'],
    neutrino_data['tau']['out_lepton_momentum_dir'], 
    neutrino_data['NC']['out_lepton_momentum_dir']
], "Out Lepton Momentum", "out_lepton_momentum_1x3")


# -------------------------------------------------------------
# Final visualizations for min_z and max_z distributions
# -------------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.hist(neutrino_data['mu']['min_z_l'], bins=80, label='numu')
plt.hist(neutrino_data['e']['min_z_l'], bins=80, label='nue')
plt.hist(neutrino_data['tau']['min_z_l'], bins=80, label='nutau')
plt.xlabel('$z_{min}$ [mm]')
plt.ylabel('Counts')
plt.yscale('log')
plt.legend(loc='upper left')
plt.savefig(f'{plot_folder}/min_z_distribution.png')
plt.show()
plt.close()

plt.figure(figsize=(8, 6))
plt.hist(neutrino_data['mu']['max_z_l'], bins=80, label='numu')
plt.hist(neutrino_data['e']['max_z_l'], bins=80, label='nue')
plt.hist(neutrino_data['tau']['max_z_l'], bins=80, label='nutau')
plt.xlabel('$z_{max}$ [mm]')
plt.ylabel('Counts')
plt.yscale('log')
plt.legend()
plt.savefig(f'{plot_folder}/max_z_distribution.png')
plt.show()
plt.close()

# -------------------------------------------------------------
# Dist_trav_lep vs dist_trav_non_lep plot
# -------------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.hist(neutrino_data['mu']['dist_trav_lep'], bins=110, range=(0,300), label='numu')
plt.hist(neutrino_data['e']['dist_trav_lep'], bins=110, range=(0,300), label='nue')
plt.hist(neutrino_data['tau']['dist_trav_lep'], bins=110, range=(0,300), label='nutau')
plt.axhline(y=800, color='white', linestyle='--', linewidth=2)
plt.xlabel('$z_{dist} = |z_{max} - z_{min}|$ [mm]')
plt.ylabel('Counts')
plt.yscale('log')
plt.legend(loc='lower right')
plt.savefig(f'{plot_folder}/dist_trav_lep_distribution.png')
plt.show()
plt.close()

# -------------------------------------------------------------
# Vox Energy for Leptons vs Non-Leptons (Energy Comparison)
# -------------------------------------------------------------

# Helper function to concatenate datasets
def concat(arrays):
    return np.concatenate(arrays, axis=0)

# -------------------------------------------------------------------------
# Plot Vox lep energy vs Vox non lep energy
# -------------------------------------------------------------------------

data_groups = [
    (neutrino_data['e']['energy_lepton_vox'], neutrino_data['e']['energy_non_lepton_vox']),
    (neutrino_data['mu']['energy_lepton_vox'], neutrino_data['mu']['energy_non_lepton_vox']),
    (neutrino_data['tau']['energy_lepton_vox'], neutrino_data['tau']['energy_non_lepton_vox']),
]


# Concatenate datasets in parallel
with ThreadPoolExecutor() as executor:
    results = list(executor.map(concat, [pair for group in data_groups for pair in group]))

# Unpack results

(all_hist_energy_lepton_vox_e, all_hist_energy_non_lepton_vox_e,
 all_hist_energy_lepton_vox_mu, all_hist_energy_non_lepton_vox_mu,
 all_hist_energy_lepton_vox_tau, all_hist_energy_non_lepton_vox_tau) = results

# Define function to plot lepton vs non-lepton energy
def plot_comparison_histograms(lepton_data, non_lepton_data, filename):
    num_bins = 100
    min_val = min(np.min(non_lepton_data), np.min(lepton_data))
    max_val = max(np.max(non_lepton_data), np.max(lepton_data))

    #if 0, skip
    if min_val == 0 or max_val == 0:
        min_val = 1e-3

    plt.hist(non_lepton_data, bins=np.logspace(np.log10(min_val), np.log10(max_val), num_bins + 1), label='Non Lepton', lw=2)
    plt.hist(lepton_data, bins=np.logspace(np.log10(min_val), np.log10(max_val), num_bins + 1), label='Primlepton', lw=2)
    plt.xlabel('Vox Energy [MeV]')
    plt.ylabel('Counts')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.savefig(f'{plot_folder}/{filename}.png')
    plt.close()

# Plot for each neutrino type
plot_comparison_histograms(all_hist_energy_lepton_vox_e, all_hist_energy_non_lepton_vox_e, 'vox_energy_distribution_nue')
plot_comparison_histograms(all_hist_energy_lepton_vox_mu, all_hist_energy_non_lepton_vox_mu, 'vox_energy_distribution_numu')
plot_comparison_histograms(all_hist_energy_lepton_vox_tau, all_hist_energy_non_lepton_vox_tau, 'vox_energy_distribution_nutau')


# -------------------------------------------------------------------------
# Summed Histograms for EM, GH, and HAD Energy
# -------------------------------------------------------------------------
data_groups = [
    (neutrino_data['e']['energy_EM_vox'], neutrino_data['e']['energy_GH_vox'], neutrino_data['e']['energy_HAD_vox']),
    (neutrino_data['mu']['energy_EM_vox'], neutrino_data['mu']['energy_GH_vox'], neutrino_data['mu']['energy_HAD_vox']),
    (neutrino_data['tau']['energy_EM_vox'], neutrino_data['tau']['energy_GH_vox'], neutrino_data['tau']['energy_HAD_vox']),
]

# Concatenate datasets in parallel
with ThreadPoolExecutor() as executor:
    results = list(executor.map(concat, [item for group in data_groups for item in group]))

# Unpack results
(all_hist_energy_EM_e, all_hist_energy_GH_e, all_hist_energy_HAD_e,
 all_hist_energy_EM_mu, all_hist_energy_GH_mu, all_hist_energy_HAD_mu,
 all_hist_energy_EM_tau, all_hist_energy_GH_tau, all_hist_energy_HAD_tau) = results

# Define function to plot summed energy distributions
def plot_energy_distribution(lepton_data, had_data, gh_data, neutrino_type):
    # Filter out non-positive values
    lepton_data = lepton_data[lepton_data > 0]
    had_data = had_data[had_data > 0]
    gh_data = gh_data[gh_data > 0]

    # Combine all data to determine global min/max for binning
    all_data = np.concatenate([lepton_data, had_data, gh_data])
    min_val = all_data.min()
    max_val = all_data.max()

    # Define log-spaced bins
    bins = np.logspace(np.log10(min_val), np.log10(max_val), 100)

    # Plot
    plt.hist(lepton_data, bins=bins, label='EM', histtype='step')
    plt.hist(had_data, bins=bins, label='HAD', histtype='step')
    plt.hist(gh_data, bins=bins, label='GH', histtype='step')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Vox Energy [MeV]')
    plt.ylabel('Counts')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{plot_folder}/energy_distributions_{neutrino_type}.png')
    plt.close()

# Plot for each neutrino type
plot_energy_distribution(all_hist_energy_EM_e, all_hist_energy_HAD_e, all_hist_energy_GH_e, 'nue')
plot_energy_distribution(all_hist_energy_EM_mu, all_hist_energy_HAD_mu, all_hist_energy_GH_mu, 'numu')
plot_energy_distribution(all_hist_energy_EM_tau, all_hist_energy_HAD_tau, all_hist_energy_GH_tau, 'nutau')


# -------------------------------------------------------------
# Saving progress and closing
# -------------------------------------------------------------
print("Processing completed.")
