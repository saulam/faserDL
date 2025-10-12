"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 05.25

Description: script to generate numpy files.
"""


from ROOT import TFile
import ROOT
import os
import glob
import numpy as np
import tqdm
import argparse
from typing import Optional, Tuple


version = "5.1"
version_reco = version + "b"
#version = version + "_tau"
#version_reco = version_reco + "_tau"
path = "/scratch2/salonso/faser/FASERCALDATA_v{}/".format(version)
true_paths = glob.glob("/scratch2/salonso/faser/FASERCALDATA_v{}/*".format(version))
reco_paths = glob.glob("/scratch2/salonso/faser/FASERCALRECODATA_v{}/*".format(version_reco))
output_dir = '/scratch/salonso/sparse-nns/faser/events_new_v{}'.format(version_reco)
os.makedirs(output_dir, exist_ok=True)
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
IS_PRIMARY = 9
IS_SECONDARY = 10
IS_TAU_DECAY = 11
IS_CHARM_DECAY = 12

MUONIC_PDGS = [-13, 13]
ELECTROMAGNETIC_PDGS = [-11, 11, -15, 15, 22]


# -----------------------------
# True hits
# -----------------------------
def get_true_hits(
    tcal_event, 
    po_event, 
    is_tau: bool, 
    is_charmed: bool
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Extract true hit information from the tcal_event and filter valid hits.

    Returns:
        (hits, hit_ids)
        hits: np.ndarray of shape (N_true, 13) with columns:
              [TRACK_ID, PARENT_ID, PRIMARY_ID, PDG, X, Y, Z, MODULE, ENERGY,
               IS_PRIMARY, IS_SECONDARY, IS_TAU_DECAY, IS_CHARM_DECAY]  (float32)
              or None if no valid hits.
        hit_ids: np.ndarray of shape (N_true,) with channel IDs (int64). Empty if no valid hits.
    """
    rows = []
    hit_ids_list = []

    # Particle objects and decay products
    particles = po_event.POs              # vector of particle objects of the event
    tau_decay = po_event.taudecay         # vector of the tau decay products
    charm_decay = po_event.charmdecay     # vector of the charm hadrons decay products

    # Build a set of GEANT track IDs for quick membership tests
    po_geant_ids = {trk.geanttrackID for trk in particles}

    # Resolve tau decay parent GEANT ID (or -1 if not applicable)
    tau_decay_geant_id = -1
    if is_tau:
        tau_parent_ids = [trk.m_trackid_in_particle[0] for trk in tau_decay if trk.nparent == 1]
        assert len(tau_parent_ids) > 0, "no tau decay products with exactly one parent found"
        assert len(set(tau_parent_ids)) == 1, "multiple tau decay parents found"
        tau_parent_id = tau_parent_ids[0]

        tau_decay_geant_id = next(
            (trk.geanttrackID for trk in particles
             if getattr(trk, "nparent", None) == 1 and trk.m_trackid_in_particle[0] == tau_parent_id),
            -1
        )
        assert tau_decay_geant_id >= 0, "tau track not found!"
    else:
        assert len(tau_decay) == 0, "tau decay tracks found in a non-nutau event!"

    # Resolve charm decay parent GEANT ID (or sentinel -1 if not applicable)
    charm_decay_geant_id = -1
    if is_charmed:
        charm_parent_ids = [trk.m_trackid_in_particle[0] for trk in charm_decay if trk.nparent == 1]
        assert len(charm_parent_ids) > 0, "no charm decay products with exactly one parent found"
        assert len(set(charm_parent_ids)) == 1, "multiple charm decay parents found"
        charm_parent_id = charm_parent_ids[0]

        charm_decay_geant_id = next(
            (trk.geanttrackID for trk in particles if trk.m_track_id == charm_parent_id),
            -1
        )
        assert charm_decay_geant_id >= 0, "charm parent not found!"
    else:
        assert len(charm_decay) == 0, "charm decay tracks found in a non-nutau event!"

    # Iterate over simulated tracks and associated hits
    for trk in tcal_event.getfTracks():
        track_id = trk.ftrackID
        parent_id = trk.fparentID
        primary_id = trk.fprimaryID
        pdg = trk.fPDG

        for hid, energy in zip(trk.fhitIDs, trk.fEnergyDeposits):
            # Keep only valid tracker channels with non-zero energy
            if tcal_event.getChannelTypefromID(hid) != 0 or energy == 0:
                continue

            # Geometry / module
            pos = tcal_event.getChannelXYZfromID(hid)
            x, y, z = pos.x(), pos.y(), pos.z()
            module = tcal_event.getChannelModulefromID(hid)

            # Flags
            is_primary_flag = (track_id == primary_id) and (parent_id == 0)
            is_secondary_flag = parent_id in po_geant_ids
            is_tau_decay_flag = parent_id == tau_decay_geant_id
            is_charm_decay_flag = parent_id == charm_decay_geant_id

            if is_primary_flag:
                assert track_id in po_geant_ids, "primary track should be present in POs"
            assert not (is_primary_flag and is_secondary_flag), "track cannot be both primary and secondary"

            rows.append([
                track_id, parent_id, primary_id, pdg,
                x, y, z, module, energy,
                is_primary_flag, is_secondary_flag, is_tau_decay_flag, is_charm_decay_flag
            ])
            hit_ids_list.append(hid)

    if not rows:
        return None, np.empty((0,), dtype=np.int64)

    hits = np.asarray(rows, dtype=np.float32)             # (N, 13) float32
    hit_ids = np.asarray(hit_ids_list, dtype=np.int64)    # (N,) int64
    return hits, hit_ids


# -----------------------------
# Mapping CSR builder
# -----------------------------
def _build_true_index_by_id(true_ids):
    """Map channel_id (int64) -> array of true hit indices (int32)."""
    index = {}
    for i, hid in enumerate(true_ids):
        index.setdefault(int(hid), []).append(i)
    for k, v in index.items():
        index[k] = np.asarray(v, dtype=np.int32)
    return index


def get_reco_hits_and_csr_map(fPORecoEvent, tcal_event, true_hits, true_ids):
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

    id_index = _build_true_index_by_id(true_ids)
    matched_lists, weight_lists = [], []

    for i, (voxel_id, psvoxel_3d) in enumerate(fPORecoEvent.PSvoxelmap):
        pos = tcal_event.getChannelXYZfromID(voxel_id)
        module = tcal_event.getChannelModulefromID(voxel_id)

        reco_hits[i, 0:3] = [pos.x(), pos.y(), pos.z()]
        reco_hits[i, 3] = module
        reco_hits[i, 4] = psvoxel_3d.RawEnergy
        reco_hits[i, 5] = float(psvoxel_3d.ghost)

        if psvoxel_3d.ghost == 0:
            ghost_mask[i] = False
            matches = id_index.get(int(voxel_id))
            if matches is None or matches.size == 0:
                reco_hits[i, 5] = 2.0
                matched_lists.append(np.empty(0, dtype=np.int32))
                weight_lists.append(np.empty(0, dtype=np.float32))
            else:
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
            reco_hits[i,5] = 1.0
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
                       out_lepton_pdg, is_cc, link_weight=None):
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


def build_pair_array(objs, dtype=None):
    """
    Build a structured NumPy array with fields:
      - g4_id
      - track_id
      - parent_id
      - pdg
    """
    if dtype is None:
        dtype = np.dtype([
            ('g4_id',    np.int32),
            ('track_id', np.int32),
            ('parent_id',np.int32),
            ('pdg',      np.int32),
        ])

    def row(o):
        parent_id = o.m_trackid_in_particle[0] if o.nparent == 1 else -1
        return (o.geanttrackID, o.m_track_id, parent_id, o.m_pdg_id)

    return np.fromiter((row(o) for o in objs), dtype=dtype)


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
    if not chunk:
        print(f"Chunk {number} is empty (len(reco_paths)={len(reco_paths)}, chunks={chunks}). Skipping.")
        return   
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
            is_cc = bool(po_event.isCC)
            is_es = bool(po_event.isES)
            is_tau = bool(po_event.istau)
            is_charmed = po_event.isCharmed()
            po          = build_pair_array(po_event.POs)          # vector of particle objets of the event
            tau_decay   = build_pair_array(po_event.taudecay)     # vector of the tau decay products
            charm_decay = build_pair_array(po_event.charmdecay)   # vector of the charm hadrons decay products
            primary_vertex = np.array([po_event.prim_vx.x(), po_event.prim_vx.y(), po_event.prim_vx.z()])
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

            # not process nutaus
            if is_cc and abs(in_neutrino_pdg) == 16 and 'tau' not in reco_file_path:
                print(f"Skipping tau neutrino event {event_id}")
                continue
            
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
            true_hits, true_ids = get_true_hits(tcal_event, po_event, is_tau, is_charmed)
            reco_hits, true_index, indptr, ghost_mask, link_weight = get_reco_hits_and_csr_map(
                tporeco_event, tcal_event, true_hits, true_ids
            )
            
            n_non_ghost = int((~ghost_mask).sum())
            if n_non_ghost < 20:
                print(f"Skipping event {event_id} with {n_non_ghost} non-ghost hits")
                continue

            seg_labels = process_labels_csr(
                true_index, indptr, ghost_mask, true_hits,
                out_lepton_pdg, is_cc, link_weight=link_weight
            )
            
            # Save event data
            np.savez_compressed(
                f'{output_dir}/{run_number}_{event_id}_{"cc" if is_cc else "nc"}_{"es" if is_es else "is"}',
                run_number=run_number,
                event_id=event_id,
                is_cc=is_cc,
                is_es=is_es,
                is_tau=is_tau,
                is_charmed=is_charmed,
                po=po,
                tau_decay=tau_decay,
                charm_decay=charm_decay,
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

        reco_file.Close()

             
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

    if not (0 <= number < chunks):
        raise ValueError(f"number must be in [0, {chunks-1}]")
  
    generate_events(number, chunks, disable)
    print("{}/{} Done!".format(number, chunks))
