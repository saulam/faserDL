"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 05.25

Description: script to generate numpy files.
"""


from ROOT import TFile
import ROOT
import glob
import numpy as np
import tqdm
import argparse


version = "5.1"
version_reco = version# + "b"
#version = version + "_tau"
#version_reco = version_reco + "_tau"
path = "/scratch2/salonso/faser/FASERCALDATA_v{}_tau3/".format(version)
true_paths = glob.glob("/scratch2/salonso/faser/FASERCALDATA_v{}_tau3/*".format(version))
reco_paths = glob.glob("/scratch2/salonso/faser/FASERCALRECODATA_v{}_tau3/*".format(version_reco))
output_dir = '/scratch/salonso/sparse-nns/faser/events_new_v{}b_tau_3'.format(version_reco)
ROOT.gSystem.Load("/scratch5/FASER/V3.1_15032025/FASER/Python_io/lib/ClassesDict.so")

# Placeholder for class objects
tcal_event = ROOT.TcalEvent()
tporeco_event = ROOT.TPORecoEvent()


# -----------------------------
# Constants for indexing arrays
# -----------------------------
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


# -----------------------------
# True hits
# -----------------------------
def get_true_hits(tcal_event):
    """
    Extracts true hit information from the tcal_event and filters valid hits.
    Returns:
        np.ndarray of shape (N_true, 9) with columns:
        [TRACK_ID, PARENT_ID, PRIMARY_ID, PDG, X, Y, Z, MODULE, ENERGY]
        or None if no valid hits.
    """
    all_hits_list = []
    for track in tcal_event.getfTracks():
        hits_info = []
        for hit_id, energy in zip(track.fhitIDs, track.fEnergyDeposits):
            if tcal_event.getChannelTypefromID(hit_id) != 0 or energy == 0:
                continue

            pos = tcal_event.getChannelXYZfromID(hit_id)
            module = tcal_event.getChannelModulefromID(hit_id)

            hits_info.append(np.array([
                track.ftrackID, track.fparentID, track.fprimaryID, track.fPDG,
                pos.x(), pos.y(), pos.z(), module, energy
            ], dtype=np.float64))

        if hits_info:
            all_hits_list.append(np.stack(hits_info))

    if not all_hits_list:
        return None

    hits = np.concatenate(all_hits_list, axis=0)

    # Cast numeric columns
    hits = hits.astype(object)
    hits[:, TRACK_ID:PARENT_ID+1] = hits[:, TRACK_ID:PARENT_ID+1].astype(np.int32)
    hits[:, PRIMARY_ID]           = hits[:, PRIMARY_ID].astype(np.int32)
    hits[:, PDG]                  = hits[:, PDG].astype(np.int32)
    hits[:, MODULE]               = hits[:, MODULE].astype(np.int16)
    hits[:, ENERGY]               = hits[:, ENERGY].astype(np.float32)
    hits[:, X_POS:Z_POS+1]        = hits[:, X_POS:Z_POS+1].astype(np.float32)
    hits = np.array(hits.tolist())
    return hits


# -----------------------------
# Mapping builder (CSR with weights)
# -----------------------------
def _build_true_spatial_index(true_hits):
    """Map (x,y,z) -> list of true hit indices."""
    if true_hits is None or true_hits.shape[0] == 0:
        return {}

    coords = true_hits[:, [X_POS, Y_POS, Z_POS]]
    keys = [tuple(row) for row in coords]
    index = {}
    for idx, key in enumerate(keys):
        index.setdefault(key, []).append(idx)
    for k, v in index.items():
        index[k] = np.asarray(v, dtype=np.int32)
    return index


def get_reco_hits_and_csr_map(fPORecoEvent, tcal_event, true_hits):
    """
    Processes reconstructed hits and builds CSR-style mapping from reco -> true.

    Returns:
        reco_hits: (N_reco, 6) float32 array [x,y,z,module,RawEnergy,ghost_flag]
        true_index: 1D int32 array of concatenated true indices
        indptr: 1D int32 array of length N_reco+1 (CSR row pointer)
        ghost_mask: 1D bool array (True if ghost or unmatched)
        link_weight: 1D float32 array, same length as true_index
                     (fraction of reco energy attributed to each true hit)
    """
    num_voxels = len(fPORecoEvent.PSvoxelmap)
    reco_hits = np.zeros((num_voxels, 6), dtype=np.float32)
    ghost_mask = np.zeros(num_voxels, dtype=bool)

    spatial_index = _build_true_spatial_index(true_hits)
    matched_lists, weight_lists = [], []

    for i, (voxel_id, psvoxel_3d) in enumerate(fPORecoEvent.PSvoxelmap):
        pos = tcal_event.getChannelXYZfromID(voxel_id)
        module = tcal_event.getChannelModulefromID(voxel_id)

        reco_hits[i, 0:3] = [pos.x(), pos.y(), pos.z()]
        reco_hits[i, 3] = module
        reco_hits[i, 4] = psvoxel_3d.RawEnergy
        reco_hits[i, 5] = float(psvoxel_3d.ghost)

        if psvoxel_3d.ghost == 0:
            key = (reco_hits[i, 0], reco_hits[i, 1], reco_hits[i, 2])
            matches = spatial_index.get(key, None)
            if matches is None or matches.size == 0:
                reco_hits[i, 5] = 2.0
                ghost_mask[i] = True
                matched_lists.append(np.empty(0, dtype=np.int32))
                weight_lists.append(np.empty(0, dtype=np.float32))
            else:
                ghost_mask[i] = False
                matched_lists.append(matches)

                # Compute per-link weights (fraction of energy)
                energies = true_hits[matches, ENERGY].astype(np.float32, copy=False)
                total = energies.sum()
                if total > 0:
                    weights = energies / total
                else:
                    weights = np.full_like(energies, 1.0 / len(energies))
                weight_lists.append(weights)
        else:
            ghost_mask[i] = True
            matched_lists.append(np.empty(0, dtype=np.int32))
            weight_lists.append(np.empty(0, dtype=np.float32))

    # Build CSR arrays
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


# -----------------------------
# Helper function
# -----------------------------
def get_matched_true_hits(i, true_hits, true_index, indptr, link_weight=None):
    """
    Retrieve the true hits (and optional weights) linked to reco voxel i.
    Returns:
        hits: (K, 9) array of true hits (empty if none)
        weights: (K,) array of weights, or None if link_weight not given
    """
    sl = slice(indptr[i], indptr[i+1])
    if sl.start == sl.stop:
        empty_hits = np.empty((0, true_hits.shape[1]), dtype=true_hits.dtype)
        empty_w = np.empty((0,), dtype=np.float32) if link_weight is not None else None
        return empty_hits, empty_w

    hits = true_hits[true_index[sl]]
    weights = link_weight[sl] if link_weight is not None else None
    return hits, weights


# -----------------------------
# Labels with per-link weights
# -----------------------------
def process_labels_csr(true_index, indptr, ghost_mask, true_hits,
                       out_lepton_pdg, is_cc, istau, link_weight=None):
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

        mu_mask   = np.isin(pdgs, MUONIC_PDGS)
        em_mask   = np.isin(pdgs, ELECTROMAGNETIC_PDGS)
        muem_mask = mu_mask | em_mask
        had_mask  = ~muem_mask

        if is_cc:
            primary_mask = (
                (matched[:, TRACK_ID].astype(np.int32) == matched[:, PRIMARY_ID].astype(np.int32)) &
                (matched[:, PARENT_ID].astype(np.int32) == 0) &
                np.isin(pdgs, out_lepton_pdg)
            )
        else:
            primary_mask = np.zeros_like(pdgs, dtype=bool)

        muem_sum = energies[muem_mask].sum() if muem_mask.any() else 0.0
        had_sum  = energies[had_mask].sum()  if had_mask.any()  else 0.0
        prim_sum = energies[primary_mask].sum() if primary_mask.any() else 0.0

        muem_primary = energies[muem_mask & primary_mask].sum() if (muem_mask & primary_mask).any() else 0.0
        had_primary  = energies[had_mask  & primary_mask].sum() if (had_mask  & primary_mask).any() else 0.0

        muem_sum -= muem_primary
        had_sum  -= had_primary

        seg_labels[i] = [0.0, muem_sum, had_sum, prim_sum]

    return seg_labels


def retrieve_hits(hits, x, y, z):
    # Returns the indices of hits that exactly match the given (x, y, z) positions
    mask = (hits[:, X_POS] == x) & (hits[:, Y_POS] == y) & (hits[:, Z_POS] == z)
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
            geom_detector = tporeco_event.geom_detector
            
            run_number, event_id = po_event.run_number, po_event.event_id
            charm = po_event.isCharmed()
            primary_vertex = np.array([po_event.prim_vx.x(), po_event.prim_vx.y(), po_event.prim_vx.z()])
            is_cc = bool(po_event.isCC)
            is_tau = bool(po_event.istau)
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
            tau_vis_momentum = np.array([po_event.tauvis_px, po_event.tauvis_py, po_event.tauvis_pz])
            tau_decay_mode = int(po_event.tau_decaymode)  # =1 e, =2 mu, =3 1-prong, =4 rho =5 3-prong, =6 other
            tau_decay_length = float(po_event.tauDecaylength())
            tau_kink_angle = float(po_event.tauKinkAngle())
            
            '''
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
            '''

            # Extract calorimeter deposited energies
            rear_cal_energy = tporeco_event.rearCals.rearCalDeposit
            rear_hcal_energy = tporeco_event.rearCals.rearHCalDeposit
            rear_mucal_energy = tporeco_event.rearCals.rearMuCalDeposit
            rear_cal_modules = np.zeros(geom_detector.rearCalNxy**2)
            rear_hcal_modules = np.zeros(geom_detector.rearHCalNxy)
            faser_cal_modules = np.zeros(10)
            for module in tporeco_event.rearCals.rearCalModule:
                rear_cal_modules[module.moduleID] = module.energyDeposit
            for module in tporeco_event.rearCals.rearHCalModule:
                rear_hcal_modules[module.moduleID] = module.energyDeposit
            for module in tporeco_event.faserCals:
                faser_cal_modules[module.ModuleID] = module.EDeposit
            rear_cal_modules = rear_cal_modules.reshape(geom_detector.rearCalNxy, geom_detector.rearCalNxy)
            faser_cal_energy = faser_cal_modules.sum()

            # Retrieve corresponding true event
            event_mask = 0
            tcal_event.Load_event(path, run_number, event_id, event_mask, po_event)

            # Extract true and reconstructed hits
            true_hits = get_true_hits(tcal_event)
            reco_hits, true_index, indptr, ghost_mask, link_weight = get_reco_hits_and_csr_map(
                tporeco_event, tcal_event, true_hits
            )
            
            if reco_hits.shape[0] - ghost_mask.sum() < 20:
                print("Skipping event {} with {} non-ghost hits".format(event_id, reco_hits.shape[0]))
                continue

            seg_labels = process_labels_csr(
                true_index, indptr, ghost_mask, true_hits,
                out_lepton_pdg, is_cc, is_tau, link_weight=link_weight
            )
            
            # Save event data
            np.savez_compressed(
                f'{output_dir}/{run_number}_{event_id}',
                run_number=run_number,
                event_id=event_id,
                is_cc=is_cc,
                is_tau=is_tau,
                charm=charm,
                e_vis=e_vis,
                sp_momentum=sp_momentum,
                vis_sp_momentum=vis_sp_momentum,
                jet_momentum=jet_momentum,
                pt_miss=pt_miss,
                primary_vertex=primary_vertex,
                true_hits=true_hits,
                reco_hits=reco_hits,
                true_index=true_index,
                indptr=indptr,
                ghost_mask=ghost_mask,
                link_weight=link_weight,
                #xz_proj = xz_proj,
                #yz_proj = yz_proj,
                #xy_projs = np.array(xy_projs, dtype=object),
                #tk_tracks=np.array(tk_tracks, dtype=object),
                #ps_tracks=ps_tracks,
                in_neutrino_pdg=in_neutrino_pdg,
                in_neutrino_momentum=in_neutrino_momentum,
                in_neutrino_energy=in_neutrino_energy,
                out_lepton_pdg=out_lepton_pdg,
                out_lepton_momentum=out_lepton_momentum,
                out_lepton_energy=out_lepton_energy,
                tau_vis_momentum=tau_vis_momentum,
                tau_decay_mode=tau_decay_mode,
                tau_decay_length=tau_decay_length,
                tau_kink_angle=tau_kink_angle,
                rear_cal_energy=rear_cal_energy,
                rear_cal_modules=rear_cal_modules,
                rear_hcal_energy=rear_hcal_energy,
                rear_hcal_modules=rear_hcal_modules,
                faser_cal_energy=faser_cal_energy,
                faser_cal_modules=faser_cal_modules,
                rear_mucal_energy=rear_mucal_energy,
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
