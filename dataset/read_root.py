"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 01.25

Description: script to generate numpy files.
"""


from ROOT import TFile, TTree, std
import ROOT
import glob
import numpy as np
import tqdm
import argparse


version = "5.1"
version_reco = version + "b"
path = "/scratch2/salonso/faser/FASERCALDATA_v{}/".format(version)
true_paths = glob.glob("/scratch2/salonso/faser/FASERCALDATA_v{}/*".format(version))
reco_paths = glob.glob("/scratch2/salonso/faser/FASERCALRECODATA_v{}/*".format(version_reco))
output_dir = '/scratch2/salonso/faser/events_v{}'.format(version_reco)

ROOT.gSystem.Load("/scratch5/FASER/V3.1_15032025/FASER/Python_io/lib/ClassesDict.so")

# Placeholder for class objects
tcal_event = ROOT.TcalEvent()
tporeco_event = ROOT.TPORecoEvent()

# Constants for indexing arrays
TRACK_ID = 0
PARENT_ID = 1
PRIMARY_ID = 2
PDG = 3
X_POS = 4
Y_POS = 5
Z_POS = 6
MODULE = 7
ENERGY = 8

MUONIC_PDGS = [-13, 13]
ELECTROMAGNETIC_PDGS = [-11, 11, -15, 15, 22]

def get_true_hits(tcal_event):
    """
    Extracts true hit information from the tcal_event and filters valid hits.
    """
    all_hits_list = []
    for track in tcal_event.getfTracks():
        hits_info = []
        for hit_id, energy in zip(track.fhitIDs, track.fEnergyDeposits):
            # Skip hits that are not of the correct type or have zero energy deposition
            if tcal_event.getChannelTypefromID(hit_id) != 0 or energy == 0:
                continue
            
            # Get hit position and module information
            position = tcal_event.getChannelXYZfromID(hit_id)
            module = tcal_event.getChannelModulefromID(hit_id)
            
            # Store hit information in an array
            hit_info = np.array([
                track.ftrackID, track.fparentID, track.fprimaryID, track.fPDG,
                position.x(), position.y(), position.z(), module, energy
            ])
            hits_info.append(hit_info)

        if hits_info:
            all_hits_list.append(np.stack(hits_info))
    
    return np.concatenate(all_hits_list) if all_hits_list else None

def get_reco_hits(fPORecoEvent, tcal_event, true_hits):
    """
    Processes reconstructed hits from the event and attempts to match them with true hits.
    """
    num_voxels = len(fPORecoEvent.PSvoxelmap)
    reco_hits = np.zeros((num_voxels, 6))
    hit_true = []

    for i, (voxel_id, psvoxel_3d) in enumerate(fPORecoEvent.PSvoxelmap):
        # Retrieve position and module information
        position = tcal_event.getChannelXYZfromID(voxel_id)
        
        # Store reconstructed hit data
        reco_hits[i] = [
            position.x(), position.y(), position.z(),
            tcal_event.getChannelModulefromID(voxel_id),
            psvoxel_3d.RawEnergy, psvoxel_3d.ghost
        ]

        if psvoxel_3d.ghost == 0:
            # Find matches within spatial threshold
            matched_hits = find_within_threshold(true_hits, *reco_hits[i, :3])
            hit_true.append(matched_hits if matched_hits.size else [-1])
            if not matched_hits.size:
                reco_hits[i, 5] = 2.  # Mark as unmatched hit
        else:
            hit_true.append([-1])
    
    return reco_hits, hit_true

def find_within_threshold(A, x, y, z):
    """
    Finds hits within a given threshold distance.
    """
    mask = (A[:, X_POS] == x) & (A[:, Y_POS] == y) & (A[:, Z_POS] == z)
    return np.where(mask)[0]

def get_tracks(tktracks):
    """
    Extracts track information including hit positions, centroid, and direction.
    """
    tracks = []
    for track in tktracks:
        centroid = np.array([track.centroid.x(), track.centroid.y(), track.centroid.z()])
        direction = np.array([track.direction.x(), track.direction.y(), track.direction.z()])
        
        # Store hits as a numpy array
        hits = np.array([[hit.point.x(), hit.point.y(), hit.point.z(), hit.eDeposit] for hit in track.tkhit])

        tracks.append({
            'hits': hits,
            'centroid': centroid,
            'direction': direction
        })
    return tracks

def contains_primary_lepton(hits, lepton_pdg, is_cc):
    """
    Checks if a primary lepton is present in the hit data (satisfies the following conditions):
        - ftrackID equals fprimaryID.
        - fparentID equals 0.
        - fPDG is equal to the PDG of the primary lepton.
    """
    if not is_cc:
        return False
    
    condition = (hits[:, TRACK_ID] == hits[:, PRIMARY_ID]) & (hits[:, PARENT_ID] == 0) & (np.isin(hits[:, PDG], lepton_pdg))
    return np.any(condition)

def process_labels(reco_hits_true, true_hits, out_lepton_pdg, is_cc):
    """
    Process a list of labels into binary classification arrays:
        - seg_labels: A (num_hits, 3) numpy array where each row represents:
            - Column 0: Ghost label (1 if ghost, 0 if not)
            - Column 1: Sum of energy depositions for muonic + electromagnetic components
            - Column 2: Sum of energy depositions for hadronic components
    
        - primlepton_labels: A (num_hits, 1) numpy array where each element is:
            - 1 if the voxel belongs to the primary lepton, 0 otherwise
    """
    num_hits = len(reco_hits_true)
    seg_labels = np.zeros((num_hits, 3))
    primlepton_labels = np.zeros((num_hits,))

    for i, reco_hit_true in enumerate(reco_hits_true):
        if reco_hit_true[0] == -1:
            # Assign ghost label
            seg_labels[i] = [1, 0, 0]
            primlepton_labels[i] = 0
            continue

        try:
            matched_hits = true_hits[reco_hit_true]
        except:
            assert False, "Not true information found for reco hits."
            
        all_pdgs = matched_hits[:, PDG]
        
        # Compute energy depositions
        m_edepo = matched_hits[np.isin(all_pdgs, MUONIC_PDGS), ENERGY].sum()
        e_edepo = matched_hits[np.isin(all_pdgs, ELECTROMAGNETIC_PDGS), ENERGY].sum()
        h_edepo = matched_hits[~np.isin(all_pdgs, list(MUONIC_PDGS) + list(ELECTROMAGNETIC_PDGS)), ENERGY].sum()
        
        primlepton_labels[i] = contains_primary_lepton(matched_hits, out_lepton_pdg, is_cc)
        seg_labels[i] = [0, m_edepo + e_edepo, h_edepo]
    
    return np.expand_dims(primlepton_labels, axis=1), seg_labels

def divide_list_into_chunks(input_list, num_chunks=1):
    """
    Divides a list into approximately equal-sized chunks.
    """
    chunk_size, remainder = divmod(len(input_list), num_chunks)
    chunks, start = [], 0
    
    for i in range(num_chunks):
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(input_list[start:end])
        start = end
    
    return chunks
   
def th2d_to_numpy(hist):
    root_array = hist.GetArray()
    n_bins_x = hist.GetNbinsX()
    n_bins_y = hist.GetNbinsY()

    # Convert the ROOT C-style array to a NumPy array, skipping the first bin (underflow)
    np_array = np.frombuffer(root_array, dtype=np.float64, count=(n_bins_y + 2) * (n_bins_x + 2))

    reshaped_array = np_array.reshape((n_bins_y + 2, n_bins_x + 2)).astype(np.float32)
    final_array = reshaped_array[1:-1, 1:-1]

    nonzero_indices = np.nonzero(final_array)
    nonzero_values = final_array[nonzero_indices]
    nonzero_coords_and_values = np.column_stack((nonzero_indices[0], nonzero_indices[1], nonzero_values))

    return nonzero_coords_and_values
 
def generate_events(number, chunks, disable):
    chunks = divide_list_into_chunks(reco_paths, num_chunks=chunks)
    chunk = chunks[number]
   
    print("First path: {}".format(chunk[0]))
    print("Number of files: {}/{}".format(len(chunk), len(reco_paths)))

    # Iterate over reconstruction files
    t = tqdm.tqdm(enumerate(chunk), total=len(chunk), disable=disable)
    for i, reco_file_path in t:
        reco_file = TFile(reco_file_path, "read")
        #reco_file.ls()
        reco_tree = reco_file["RecoEvent"]
        #reco_tree.Print()
        total_entries = reco_tree.GetEntries()
        reco_tree.SetBranchAddress("TPORecoEvent", tporeco_event)
    
        for entry_idx in range(total_entries):
            reco_tree.GetEntry(entry_idx)

            # Extract global event information
            po_event = tporeco_event.GetPOEvent()
            run_number, event_id = po_event.run_number, po_event.event_id
            primary_vertex = np.array([po_event.prim_vx.x(), po_event.prim_vx.y(), po_event.prim_vx.z()])
            is_cc = bool(po_event.isCC)
            e_vis = po_event.Evis
            sp_momentum = np.array([po_event.spx, po_event.spy, po_event.spz])
            vis_sp_momentum = np.array([po_event.vis_spx, po_event.vis_spy, po_event.vis_spz])
            jet_momentum = np.array([po_event.jetpx, po_event.jetpy, po_event.jetpz])
            pt_miss = po_event.ptmiss
            in_neutrino, out_lepton = po_event.in_neutrino, po_event.out_lepton
            in_neutrino_pdg = in_neutrino.m_pdg_id
            in_neutrino_momentum = np.array([in_neutrino.m_px, in_neutrino.m_py, in_neutrino.m_pz])
            in_neutrino_energy = in_neutrino.m_energy
            out_lepton_pdg = out_lepton.m_pdg_id
            out_lepton_momentum = np.array([out_lepton.m_px, out_lepton.m_py, out_lepton.m_pz])
            out_lepton_energy = out_lepton.m_energy

            # Extract views
            xz_view = tporeco_event.Get2DViewXPS()
            yz_view = tporeco_event.Get2DViewYPS()
            z_view = tporeco_event.zviewPS
            xz_proj = th2d_to_numpy(xz_view)
            yz_proj = th2d_to_numpy(yz_view)
            xy_projs = []
            for layer in range(len(z_view)):
                view = z_view[layer]
                proj = th2d_to_numpy(view)
                xy_projs.append(proj)

            # Extract track data
            tk_tracks = get_tracks(tporeco_event.fTKTracks)
            ps_tracks = get_tracks(tporeco_event.fPSTracks)

            # Extract calorimeter deposited energies
            rear_cal_energy = tporeco_event.rearCals.rearCalDeposit
            rear_hcal_energy = tporeco_event.rearCals.rearHCalDeposit
            rear_mucal_energy = tporeco_event.rearCals.rearMuCalDeposit
            rear_hcal_modules = np.zeros(9)
            faser_cal_modules = np.zeros(10) 
            for module in tporeco_event.rearCals.rearHCalModule:
                rear_hcal_modules[module.moduleID] = module.energyDeposit
            for module in tporeco_event.faserCals:
                faser_cal_modules[module.ModuleID] = module.EDeposit
            faser_cal_energy = faser_cal_modules.sum()

            # Retrieve corresponding true event
            event_mask = 0
            tcal_event.Load_event(path, run_number, event_id, event_mask, po_event)

            # Extract true and reconstructed hits
            true_hits = get_true_hits(tcal_event)
            reco_hits, reco_hits_true = get_reco_hits(tporeco_event, tcal_event, true_hits)

            primlepton_labels, seg_labels = process_labels(reco_hits_true, true_hits, out_lepton_pdg, is_cc)
            
            # Save event data
            np.savez_compressed(
                f'{output_dir}/{run_number}_{event_id}',
                run_number=run_number,
                event_id=event_id,
                is_cc=is_cc,
                e_vis=e_vis,
                sp_momentum=sp_momentum,
                vis_sp_momentum=vis_sp_momentum,
                jet_momentum=jet_momentum,
                pt_miss=pt_miss,
                primary_vertex=primary_vertex,
                true_hits=true_hits,
                reco_hits=reco_hits,
                reco_hits_true=np.array(reco_hits_true, dtype=object),
                xz_proj = xz_proj,
                yz_proj = yz_proj,
                xy_projs = np.array(xy_projs, dtype=object),
                tk_tracks=np.array(tk_tracks, dtype=object),
                ps_tracks=ps_tracks,
                in_neutrino_pdg=in_neutrino_pdg,
                in_neutrino_momentum=in_neutrino_momentum,
                in_neutrino_energy=in_neutrino_energy,
                out_lepton_pdg=out_lepton_pdg,
                out_lepton_momentum=out_lepton_momentum,
                out_lepton_energy=out_lepton_energy,
                rear_cal_energy=rear_cal_energy,
                rear_hcal_modules=rear_hcal_modules,
                rear_hcal_energy=rear_hcal_energy,
                rear_mucal_energy=rear_mucal_energy,
                faser_cal_modules=faser_cal_modules,
                faser_cal_energy=faser_cal_energy,
                primlepton_labels=primlepton_labels,
                seg_labels=seg_labels,
            )

             
# Main function to handle command-line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Divide a list into chunks and select one chunk based on a number.")
    parser.add_argument('--number', type=int, required=True, help="Number (0-9) to select the chunk")
    parser.add_argument('--chunks', type=int, required=True, help="Number of chunks")
    parser.add_argument("--disable", action="store_true", default=False, help="Disable progressbar")
    args = parser.parse_args()
    number = args.number
    chunks = args.chunks
    disable = args.disable

    # Validate number range
    if number < 0 or number >= chunks:
        raise ValueError("Number must be between 0 and 9")
  
    generate_events(number, chunks, disable)
    print("{}/{} Done!".format(number, chunks))

