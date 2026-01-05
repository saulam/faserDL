"""
Authors: Dr. Saul Alonso-Monsalve, Fabio Cufino
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 12.25

Description: script to generate numpy files.
"""

from ROOT import TFile
import ROOT
import os
import glob
import numpy as np
import tqdm
import argparse
import sys
from typing import Optional, Tuple
from array import array
from collections import defaultdict

# =============================================================================
# CONFIGURATION AND SETUP
# =============================================================================

# --- Load Libraries ---
ROOT.gSystem.Load("/scratch/fcufino/FASER/GenFit-install/lib/libgenfit2.so")  
ROOT.gSystem.Load("/scratch/fcufino/FASER/rave-install/lib/libRaveBase.so")

geo_path = "/scratch/fcufino/FASER/GeomGDML/geometry_tilted_5degree.gdml"
dict_path = "/scratch/fcufino/FASER/Python_io/lib/ClassesDict.so"

status = ROOT.gSystem.Load(dict_path)
if status < 0:
    print(f"ERROR: Could not load dictionary at {dict_path}")
    sys.exit(1)

# --- Load Geometry ---
if os.path.exists(geo_path):
    ROOT.TGeoManager.Import(geo_path)
else:
    print(f"WARNING: Geometry file not found at {geo_path}")

# --- FASER Environment Setup ---
VERSION = "7.0_tau"
BASE_PATH = "/scratch2/salonso/faser/"
CAL_DIR = f"{BASE_PATH}FASERCALDATA_v{VERSION}/" + "{}/"
RECO_DIR = f"{BASE_PATH}FASERCALRECODATA_v{VERSION}*"
OUTPUT_DIR = f'/scratch/salonso/sparse-nns/faser/events_v{VERSION}_npz'

# Validate input directories
tcal_path = glob.glob(CAL_DIR.format("*", "*") + "*.root")
if not tcal_path:
    print(f"Error: No CAL/Truth ROOT files found in {CAL_DIR}")
    sys.exit(1)

reco_paths = (
    glob.glob(os.path.join(RECO_DIR, "*.root")) +
    glob.glob(os.path.join(RECO_DIR, "*", "*.root"))
)

print(f'Saving data in: {OUTPUT_DIR}')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Placeholder for class objects
tcal_event = ROOT.TcalEvent()
tporeco_event = ROOT.TPORecoEvent()

# =============================================================================
# CONSTANTS
# =============================================================================

# Array indexing constants
TRACK_ID = 0
PARENT_ID = 1
PRIMARY_ID = 2
PDG = 3
X_POS = 4
Y_POS = 5
Z_POS = 6
MODULE = 7
ENERGY = 8
IS_PRIMARY = 9
IS_SECONDARY = 10
IS_TAU_DECAY = 11
IS_CHARM_DECAY = 12

# PDG codes
MUONIC_PDGS = [-13, 13]
ELECTROMAGNETIC_PDGS = [-11, 11, -15, 15, 22]

# Processing constants
MAXMUTRACKS = 10
N_LAYERS_Z = 20

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_discrete_coord_from_id(ID, N_LAYERS_Z=20):
    """
    Decode a channel ID into discrete voxel coordinates (x, y, z).

    Works without geometry by unpacking the encoded ID.
    Returns a flattened z index combining layer and intra-layer depth.
    """

    hittype = ID // 100_000_000_000

    # --------------------------------------------------
    # Scintillator hit
    # --------------------------------------------------
    if hittype == 0:
        ix = ID % 1000
        iy = (ID // 1000) % 1000
        iz = (ID // 1_000_000) % 1000
        ilayer = ID // 1_000_000_000

        # 0-based flattened z index
        z = iz + ilayer * N_LAYERS_Z

        return ix, iy, z

    # --------------------------------------------------
    # Silicon tracker hit
    # --------------------------------------------------
    elif hittype == 1:
        ix = ID % 10000
        iy = (ID // 10000) % 10000
        ilayer = (ID // 100_000_000) % 100
        icopy = (ID // 10_000_000_000) % 10

        # two silicon planes per layer (copy 0 / 1)
        z = ilayer * 2 + icopy

        return ix, iy, z

    # --------------------------------------------------
    # Unknown hit type
    # --------------------------------------------------
    else:
        raise ValueError(f"Unknown hittype {hittype}")


def get_true_hits(tcal_event, po_event, is_tau: bool, is_charmed: bool):
    """
    Extract true hit information from the tcal_event and filter valid hits.

    Returns:
        (hits, hit_ids)
        hits: np.ndarray of shape (N_true, 13) with columns:
              [TRACK_ID, PARENT_ID, PRIMARY_ID, PDG, X, Y, Z, MODULE, ENERGY,
               IS_PRIMARY, IS_SECONDARY, IS_TAU_DECAY, IS_CHARM_DECAY]  (float32)
              or None if no valid hits.
              - X,Y,Z are discrete voxel coordinates, MODULE is the module ID,
        hit_ids: np.ndarray of shape (N_true,) with channel IDs (int64). Empty if no valid hits.
    """
    rows, hit_ids_list = [], []
    particles = po_event.POs
    tau_decay = po_event.taudecay
    charm_decay = po_event.charmdecay
    
    # Pre-calculate IDs
    po_geant_ids = {trk.geanttrackID for trk in particles}

    # Find tau decay GEANT ID
    tau_decay_geant_id = -1
    if is_tau:
        tau_parent_ids = [trk.m_trackid_in_particle[0] for trk in tau_decay if trk.nparent == 1]
        if tau_parent_ids:
            tau_parent_id = tau_parent_ids[0]
            tau_decay_geant_id = next(
                (trk.geanttrackID for trk in particles
                 if getattr(trk, "nparent", None) == 1 and trk.m_trackid_in_particle[0] == tau_parent_id), -1)

    # Find charm decay GEANT ID
    charm_decay_geant_id = -1
    if is_charmed:
        charm_parent_ids = [trk.m_trackid_in_particle[0] for trk in charm_decay if trk.nparent == 1]
        if charm_parent_ids:
            charm_parent_id = charm_parent_ids[0]
            charm_decay_geant_id = next(
                (trk.geanttrackID for trk in particles if trk.m_track_id == charm_parent_id), -1)

    # Process tracks and hits
    for trk in tcal_event.getfTracks():
        track_id = trk.ftrackID
        parent_id = trk.fparentID
        primary_id = trk.fprimaryID
        pdg = trk.fPDG

        for hid, energy in zip(trk.fhitIDs, trk.fEnergyDeposits):
            if tcal_event.getChannelTypefromID(hid) != 0 or energy == 0:
                continue

            x, y, z = get_discrete_coord_from_id(hid, N_LAYERS_Z=N_LAYERS_Z)
            module = tcal_event.getChannelModulefromID(hid)

            is_primary_flag = (track_id == primary_id) and (parent_id == 0)
            is_secondary_flag = parent_id in po_geant_ids
            is_tau_decay_flag = parent_id == tau_decay_geant_id
            is_charm_decay_flag = parent_id == charm_decay_geant_id

            rows.append([
                track_id, parent_id, primary_id, pdg,
                x, y, z, module, energy,
                is_primary_flag, is_secondary_flag, is_tau_decay_flag, is_charm_decay_flag
            ])
            hit_ids_list.append(hid)

    if not rows:
        return None, np.empty((0,), dtype=np.int64)

    hits = np.asarray(rows, dtype=np.float32)
    hit_ids = np.asarray(hit_ids_list, dtype=np.int64)
    return hits, hit_ids


def _build_true_index_by_id(true_ids):
    """Build an index mapping hit IDs to their positions in the true_hits array."""
    index = {}
    for i, hid in enumerate(true_ids):
        index.setdefault(int(hid), []).append(i)
    for k, v in index.items():
        index[k] = np.asarray(v, dtype=np.int32)
    return index


def get_reco_hits_and_csr_map(fPORecoEvent, geom_helper, true_hits, true_ids):
    """
    Processes reconstructed hits and builds CSR-style mapping from reco -> true.

    Returns:
        reco_hits: (N_reco, 6) float32 array [x,y,z,module,RawEnergy,ghost_flag]
            - x, y, z are discrete voxel coordinates
        true_index: 1D int32 array of concatenated true indices
        indptr: 1D int32 array of length N_reco+1 (CSR row pointer)
        ghost_mask: 1D bool array (True if ghost or unmatched)
        link_weight: 1D float32 array, same length as true_index
                     (fraction of reco energy attributed to each true hit)
    """
    num_voxels = len(fPORecoEvent.PSvoxelmap)
    reco_hits = np.zeros((num_voxels, 6), dtype=np.float32)
    ghost_mask = np.zeros(num_voxels, dtype=bool)
    id_index = _build_true_index_by_id(true_ids)
    matched_lists, weight_lists = [], []

    for i, (voxel_id, psvoxel_3d) in enumerate(fPORecoEvent.PSvoxelmap):
        x, y, z = get_discrete_coord_from_id(voxel_id, N_LAYERS_Z=N_LAYERS_Z)
        module = geom_helper.getChannelModulefromID(voxel_id)

        reco_hits[i, 0:3] = [x, y, z]
        reco_hits[i, 3] = module
        reco_hits[i, 4] = psvoxel_3d.RawEnergy
        reco_hits[i, 5] = float(psvoxel_3d.ghost)


        if psvoxel_3d.ghost == 0:
            matches = id_index.get(int(voxel_id))
            if matches is None or matches.size == 0:
                reco_hits[i, 5] = 2.0
                matched_lists.append(np.empty(0, dtype=np.int32))
                weight_lists.append(np.empty(0, dtype=np.float32))
            else:
                matched_lists.append(matches)
                energies = true_hits[matches, ENERGY].astype(np.float32, copy=False)
                total = energies.sum()
                weights = energies / total if total > 0 else np.full_like(energies, 1.0 / len(energies))
                weight_lists.append(weights)
        else:
            ghost_mask[i] = True
            matched_lists.append(np.empty(0, dtype=np.int32))
            weight_lists.append(np.empty(0, dtype=np.float32))

    # Build CSR structure
    counts = np.fromiter((m.size for m in matched_lists), count=num_voxels, dtype=np.int32)
    indptr = np.empty(num_voxels + 1, dtype=np.int32)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    total = int(indptr[-1])
    true_index = np.empty(total, dtype=np.int32)
    link_weight = np.empty(total, dtype=np.float32)

    offset = 0
    for m, w in zip(matched_lists, weight_lists):
        n = m.size
        if n:
            true_index[offset:offset+n] = m
            link_weight[offset:offset+n] = w
        offset += n

    return reco_hits, true_index, indptr, ghost_mask, link_weight


def process_labels_csr(true_index, indptr, ghost_mask, true_hits, out_lepton_pdg, is_cc, link_weight=None):
    """
    Computes seg_labels with optional weighted contributions.

    Returns:
        seg_labels: (num_hits, 4) float32
            [:,0] -> Ghost label (1 if ghost/unmatched, 0 otherwise)
            [:,1] -> (muonic + electromagnetic) E_dep, minus primary-lepton E
            [:,2] -> hadronic E_dep, minus primary-lepton E
            [:,3] -> primary-lepton E_dep
    """
    num_hits = indptr.size - 1
    seg_labels = np.zeros((num_hits, 4), dtype=np.float32)
    no_true = (true_hits is None) or (true_hits.shape[0] == 0)

    for i in range(num_hits):
        if ghost_mask[i] or no_true:
            seg_labels[i] = [1.0, 0.0, 0.0, 0.0]
            continue
        
        sl = slice(indptr[i], indptr[i+1])
        if sl.start == sl.stop:
            seg_labels[i] = [1.0, 0.0, 0.0, 0.0]
            continue
        
        matched = true_hits[true_index[sl]]
        pdgs = matched[:, PDG].astype(np.int32, copy=False)
        energies = matched[:, ENERGY].astype(np.float32, copy=False)
        if link_weight is not None:
            energies = energies * link_weight[sl]

        mu_mask = np.isin(pdgs, MUONIC_PDGS)
        em_mask = np.isin(pdgs, ELECTROMAGNETIC_PDGS)
        muem_mask = mu_mask | em_mask
        had_mask = ~muem_mask

        primary_mask = (
            (matched[:, TRACK_ID].astype(np.int32) == matched[:, PRIMARY_ID].astype(np.int32)) &
            (matched[:, PARENT_ID].astype(np.int32) == 0) &
            np.isin(pdgs, out_lepton_pdg)
        ) if is_cc else np.zeros_like(pdgs, dtype=bool)

        muem_sum = energies[muem_mask].sum() - energies[muem_mask & primary_mask].sum()
        had_sum = energies[had_mask].sum() - energies[had_mask & primary_mask].sum()
        prim_sum = energies[primary_mask].sum()

        seg_labels[i] = [0.0, muem_sum, had_sum, prim_sum]
    
    return seg_labels


def get_muon_spectrometer(fMuTracks):
    """Extract muon spectrometer track information."""
    ntracks = min(int(len(fMuTracks)), MAXMUTRACKS)
    # Safely slice to ntracks and iterate once
    rows = []
    for track in fMuTracks[:ntracks]:
        rows.append([track.fcharge, track.fpos.size(), track.fpx, track.fpy, track.fpz,
                     track.fp, track.fchi2, track.fnDoF, track.fpval, track.fpErr, track.fipErr])
    rows = np.array(rows, dtype=np.float32).T
    return ntracks, rows


def build_pair_array(objs, dtype=None):
    """Build a structured NumPy array with fields: g4_id, track_id, parent_id, pdg."""
    if dtype is None:
        dtype = np.dtype([
            ('g4_id', np.int32),
            ('track_id', np.int32),
            ('parent_id', np.int32),
            ('pdg', np.int32),
        ])

    def row(o):
        parent_id = o.m_trackid_in_particle[0] if o.nparent == 1 else -1
        return (o.geanttrackID, o.m_track_id, parent_id, o.m_pdg_id)

    return np.fromiter((row(o) for o in objs), dtype=dtype)


def getChannelXYZRearHCal(moduleID):
    """Extract XYZ coordinates from rear HCal module ID."""
    x = moduleID % 1000
    y = (moduleID // 1000) % 1000
    z = (moduleID // 1000000) % 1000
    return x, y, z


def th2d_to_numpy(hist):
    """Convert ROOT TH2D histogram to NumPy array of non-zero values."""
    root_array = hist.GetArray()
    n_bins_x = hist.GetNbinsX()
    n_bins_y = hist.GetNbinsY()

    np_array = np.frombuffer(root_array, dtype=np.float64, count=(n_bins_y + 2) * (n_bins_x + 2))
    reshaped_array = np_array.reshape((n_bins_y + 2, n_bins_x + 2)).astype(np.float32)
    final_array = reshaped_array[1:-1, 1:-1]

    nonzero_indices = np.nonzero(final_array)
    nonzero_values = final_array[nonzero_indices]
    nonzero_coords_and_values = np.column_stack((nonzero_indices[0], nonzero_indices[1], nonzero_values))

    return nonzero_coords_and_values


def get_tracks(tktracks):
    """Extract track information including hit positions, centroid, and direction."""
    tracks = []
    for track in tktracks:
        centroid = np.array([track.centroid.x(), track.centroid.y(), track.centroid.z()])
        direction = np.array([track.direction.x(), track.direction.y(), track.direction.z()])
        hits = np.array([[hit.point.x(), hit.point.y(), hit.point.z(), hit.eDeposit] for hit in track.tkhit])

        tracks.append({
            'hits': hits,
            'centroid': centroid,
            'direction': direction
        })
    return tracks


def divide_list_into_chunks(input_list, num_chunks=1):
    """Divide a list into approximately equal chunks."""
    chunk_size, remainder = divmod(len(input_list), num_chunks)
    chunks, start = [], 0
    
    for i in range(num_chunks):
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(input_list[start:end])
        start = end
    
    return chunks

# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================

def generate_events(number, chunks, disable):
    """Process a chunk of RECO files and generate event data."""
    
    print(f"--- File Discovery Check ---")
    print(f"Total RECO files found: {len(reco_paths)}")
    print(f"CAL/Truth files expected in directory: {CAL_DIR}")
    if reco_paths:
        print(f"First RECO file found: {reco_paths[0]}")
    else:
        print(f"First RECO file found: NONE")
    print(f"--- End Check ---")

    # Select the current chunk
    all_chunks = divide_list_into_chunks(reco_paths, num_chunks=chunks)
    if number >= len(all_chunks):
        print(f"Error: Chunk number {number} is out of range for {len(all_chunks)} chunks.")
        return

    chunk = all_chunks[number]
    if not chunk:
        print(f"Chunk {number} is empty. Skipping.")
        return   
    
    print(f"First file in chunk: {chunk[0]}")
    print(f"Number of files in chunk: {len(chunk)} / {len(reco_paths)}")

    # Iterate over reconstruction files in the assigned chunk
    t = tqdm.tqdm(enumerate(chunk), total=len(chunk), disable=disable)
    events_saved_in_chunk = 0

    for i, reco_file_path in t:
        reco_file = TFile(reco_file_path, "read")
        reco_tree = reco_file["RecoEvent"]
        reco_tree.SetBranchAddress("TPORecoEvent", tporeco_event)
        subfolder = os.path.basename(os.path.dirname(reco_file_path))
        total_entries = reco_tree.GetEntries()
    
        for entry_idx in range(total_entries):
            reco_tree.GetEntry(entry_idx)

            po_event = tporeco_event.GetPOEvent()
            geom_detector = tporeco_event.geom_detector
            
            # Extract event metadata
            run_number, event_id = po_event.run_number, po_event.event_id
            is_cc = bool(po_event.isCC)
            is_es = bool(po_event.isES())
            is_tau = bool(po_event.istau)
            is_charmed = bool(po_event.isCharmed())
            
            # Extract particle objects
            po = build_pair_array(po_event.POs)
            tau_decay = build_pair_array(po_event.taudecay)
            charm_decay = build_pair_array(po_event.charmdecay)
            
            # Extract event kinematics
            primary_vertex = np.array([po_event.prim_vx.x(), po_event.prim_vx.y(), po_event.prim_vx.z()])
            e_vis = po_event.Evis
            sp_momentum = np.array([po_event.spx, po_event.spy, po_event.spz])
            vis_sp_momentum = np.array([po_event.vis_spx, po_event.vis_spy, po_event.vis_spz])
            jet_momentum = np.array([po_event.jetpx, po_event.jetpy, po_event.jetpz])
            pt_miss = po_event.ptmiss
            
            # Extract neutrino and lepton information
            in_neutrino, out_lepton = po_event.in_neutrino, po_event.out_lepton
            in_neutrino_pdg = in_neutrino.m_pdg_id
            in_neutrino_momentum = np.array([in_neutrino.m_px, in_neutrino.m_py, in_neutrino.m_pz])
            in_neutrino_energy = in_neutrino.m_energy
            out_lepton_pdg = out_lepton.m_pdg_id
            out_lepton_momentum = np.array([out_lepton.m_px, out_lepton.m_py, out_lepton.m_pz])
            out_lepton_energy = out_lepton.m_energy
            
            # Extract tau-specific information
            tau_vis_momentum = np.array([po_event.tauvis_px, po_event.tauvis_py, po_event.tauvis_pz])
            tau_decay_mode = int(po_event.tau_decaymode)
            tau_decay_length = float(po_event.tauDecaylength())
            tau_kink_angle = float(po_event.tauKinkAngle())

            # Extract 2D views (COMMENTED OUT - only enable if needed)
            # xz_view = tporeco_event.Get2DViewXPS()
            # yz_view = tporeco_event.Get2DViewYPS()
            # z_view = tporeco_event.zviewPS
            # xz_proj = th2d_to_numpy(xz_view)
            # yz_proj = th2d_to_numpy(yz_view)
            # xy_projs = [th2d_to_numpy(z_view[layer]) for layer in range(len(z_view))]

            # Extract track data (COMMENTED OUT - only enable if needed)
            # tk_tracks = get_tracks(tporeco_event.fTKTracks)
            # ps_tracks = get_tracks(tporeco_event.fPSTracks)
        
            # Load TcalEvent
            event_mask = 0
            tcal_path = os.path.dirname(reco_file_path.replace("FASERCALRECODATA", "FASERCALDATA")) + "/"
            tcal_event.Load_event(tcal_path, run_number, event_id, event_mask, po_event)

            # Extract rear calorimeter hits
            ecal_hits = np.zeros(geom_detector.rearCalNxy**2)
            for x in tcal_event.rearCalDeposit:
                ecal_hits[x.moduleID] = x.energyDeposit
            ecal_hits = ecal_hits.reshape(geom_detector.rearCalNxy, geom_detector.rearCalNxy)

            ahcal_hits = np.zeros(shape=(len(tcal_event.rearHCalDeposit), 4))
            for i, x in enumerate(tcal_event.rearHCalDeposit):
                ahcal_hits[i, :3] = getChannelXYZRearHCal(x.moduleID)
                ahcal_hits[i, 3] = x.energyDeposit

            # Extract true and reconstructed hits
            true_hits, true_ids = get_true_hits(tcal_event, po_event, is_tau, is_charmed)
            reco_hits, true_index, indptr, ghost_mask, link_weight = get_reco_hits_and_csr_map(
                tporeco_event, tcal_event, true_hits, true_ids
            )
            
            # Filter events with insufficient hits
            n_non_ghost = int((~ghost_mask).sum())
            if n_non_ghost < 20:
                continue

            # Process segmentation labels
            seg_labels = process_labels_csr(
                true_index, indptr, ghost_mask, true_hits,
                out_lepton_pdg, is_cc, link_weight=link_weight
            )

            # Extract muon spectrometer data
            muspec_ntracks, muspec_info = get_muon_spectrometer(tporeco_event.fMuTracks)

            # Package event data
            event_data = {
                'run_number': run_number,
                'event_id': event_id,
                'is_cc': is_cc,
                'is_es': is_es,
                'is_tau': is_tau,
                'is_charmed': is_charmed,
                'po': po,
                'tau_decay': tau_decay,
                'charm_decay': charm_decay,
                'e_vis': e_vis,
                'sp_momentum': sp_momentum,
                'vis_sp_momentum': vis_sp_momentum,
                'jet_momentum': jet_momentum,
                'pt_miss': pt_miss,
                'primary_vertex': primary_vertex,
                'true_hits': true_hits,
                'reco_hits': reco_hits,
                'true_index': true_index,
                'indptr': indptr,
                'ghost_mask': ghost_mask,
                'link_weight': link_weight,
                'in_neutrino_pdg': in_neutrino_pdg,
                'in_neutrino_momentum': in_neutrino_momentum,
                'in_neutrino_energy': in_neutrino_energy,
                'out_lepton_pdg': out_lepton_pdg,
                'out_lepton_momentum': out_lepton_momentum,
                'out_lepton_energy': out_lepton_energy,
                'tau_vis_momentum': tau_vis_momentum,
                'tau_decay_mode': tau_decay_mode,
                'tau_decay_length': tau_decay_length,
                'tau_kink_angle': tau_kink_angle,
                'ecal_hits': ecal_hits,
                'ahcal_hits': ahcal_hits,
                'seg_labels': seg_labels,
                'muspec_ntracks': muspec_ntracks,
                'muspec_info': muspec_info,
            }

            # Save the single event data
            output_filename = f'{OUTPUT_DIR}/run_{run_number}_event_{event_id}.npz'
            np.savez_compressed(output_filename, **event_data)

            events_saved_in_chunk += 1
            t.set_description(f"File: {os.path.basename(reco_file_path)} | Saved (Run:{run_number}, Event:{event_id})")

        reco_file.Close()

    print(f"\nSuccessfully processed chunk {number}/{chunks}.")
    print(f"Total events saved in chunk {number}: {events_saved_in_chunk}")
    print(f"Individual NPZ files are in: {OUTPUT_DIR}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process FASER events in parallel chunks.")
    parser.add_argument('--number', type=int, required=True, help="Chunk number to process (0-based)")
    parser.add_argument('--chunks', type=int, required=True, help="Total number of chunks")
    parser.add_argument("--disable", action="store_true", default=False, help="Disable progress bar")
    args = parser.parse_args()

    if not (0 <= args.number < args.chunks):
        raise ValueError(f"number must be in [0, {args.chunks-1}]")
  
    generate_events(args.number, args.chunks, args.disable)
    print(f"{args.number}/{args.chunks} Done!")