# needed packages
import argparse
import tqdm
import io
import os
import sys
from contextlib import redirect_stdout, redirect_stderr
import numpy as np
import glob
import ROOT
import tqdm
from ROOT import TFile
from ROOT import std
import matplotlib.pyplot as plt

def get_true_hits(tcal_event):
    all_hits_list = []
    tracks = tcal_event.getfTracks()
    for track in tracks:
        nhits = track.fhitIDs.size()
        hits_info = []
        for i in range(nhits):
            hittype = tcal_event.getChannelTypefromID(track.fhitIDs[i])
            #if (hittype != 0 or track.fEnergyDeposits[i] < 0.5):
            if (hittype != 0 or track.fEnergyDeposits[i] == 0):
                continue
            
            position = tcal_event.getChannelXYZfromID(track.fhitIDs[i])
            module = tcal_event.getChannelModulefromID(track.fhitIDs[i])
            hit_info = np.zeros(shape=(9,))
            hit_info[0] = track.ftrackID
            hit_info[1] = track.fparentID
            hit_info[2] = track.fprimaryID
            hit_info[3] = track.fPDG
            hit_info[4] = position.x()
            hit_info[5] = position.y()
            hit_info[6] = position.z()
            hit_info[7] = module
            hit_info[8] = track.fEnergyDeposits[i]
            hits_info.append(hit_info)

        if len(hits_info) > 0:
            hits_info = np.stack(hits_info)
            all_hits_list.append(hits_info)

    if len(all_hits_list) > 0:
        all_hits = np.concatenate(all_hits_list)
    else:
        all_hits = None

    return all_hits

def get_reco_hits(fPORecoEvent, tcal_event, true_hits, geom_detector):
    reco_hits = np.zeros(shape=(len(fPORecoEvent.PSvoxelmap), 6))
    hit_true = []
    for i, (voxel_id, psvoxel_3d) in enumerate(fPORecoEvent.PSvoxelmap):
        voxel_id = psvoxel_3d.ID
        raw_energy = psvoxel_3d.RawEnergy  # MeV
        ghost = psvoxel_3d.ghost            
        module = tcal_event.getChannelModulefromID(voxel_id)
        position = tcal_event.getChannelXYZfromID(voxel_id)
        x, y, z = position.x(), position.y(), position.z()
        
        reco_hits[i, 0] = x
        reco_hits[i, 1] = y
        reco_hits[i, 2] = z
        reco_hits[i, 3] = module
        reco_hits[i, 4] = raw_energy
        reco_hits[i, 5] = ghost
           
        if ghost == 0:
            matched_hits = find_within_threshold(true_hits, x, y, z, geom_detector)
            if len(matched_hits) == 0:
                # if track hit, we should be able to find some truth
                reco_hits[i, 5] = 2.
                hit_true.append([-1])
                continue
            hit_true.append(matched_hits)
        else:
            hit_true.append([-1])
        
    return reco_hits, hit_true

def find_within_threshold(A, x, y, z, geom_detector):
    threshold = geom_detector.fScintillatorVoxelSize
    mask = (
        #((A[:, 4] - x) >= 0) &
        #((A[:, 4] - x) < threshold) &
        #((A[:, 5] - y) >= 0) &
        #((A[:, 5] - y) < threshold) &
        #((A[:, 6] - z) >= 0) &
        #((A[:, 6] - z) < threshold)
        (A[:, 4] == x) & (A[:, 5] == y) & (A[:, 6] == z)
    )
    return np.where(mask)[0]

def th2d_to_numpy(hist):
    root_array = hist.GetArray()
    # Get the number of bins along X and Y
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
    # Calculate the chunk size
    chunk_size = len(input_list) // num_chunks
    # Remainder for uneven chunks
    remainder = len(input_list) % num_chunks
    
    # Create the chunks
    chunks = []
    start = 0
    for i in range(num_chunks):
        # Add one more element to some chunks if remainder is not zero
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(input_list[start:end])
        start = end
    
    return chunks

def get_tracks(tktracks):
    tracks = []
    for track in tktracks:
        tkhits = track.tkhit
        nhits = len(tkhits)
        centroid = track.centroid
        direction = track.direction
        centroid = np.array([centroid.x(), centroid.y(), centroid.z()])
        direction = np.array([direction.x(), direction.y(), direction.z()])

        hits = np.zeros(shape=(nhits, 4))
        for i, hit in enumerate(tkhits):
            hits[i, 0] = hit.point.x()
            hits[i, 1] = hit.point.y()
            hits[i, 2] = hit.point.z()
            hits[i, 3] = hit.eDeposit

        tktrack = {'hits': hits, 'centroid': centroid, 'direction': direction}
        tracks.append(tktrack)
    return tracks

def generate_events(number, chunks, disable):
    path = "/scratch2/salonso/faser/FASERCALDATA_v3.1/"
    ROOT.gSystem.Load("/scratch5/FASER/Python_io/lib/ClassesDict.so")

    files = glob.glob(path + "*.root")
    chunks = divide_list_into_chunks(files, num_chunks=chunks)

    chunk = chunks[number]
   
    print("First path: {}".format(chunk[0]))
    print("Number of files: {}/{}".format(len(chunk), len(files)))
    t = tqdm.tqdm(enumerate(chunk), total=len(chunk), disable=disable)
    for i, file in t:
        #if i < 7600:
        #    continue
        with open(os.devnull, 'w') as devnull:
            old_stdout_fno = os.dup(1)  # Save the current stdout file descriptor
            old_stderr_fno = os.dup(2)  # Save the current stderr file descriptor
            os.dup2(devnull.fileno(), 1)  # Redirect stdout to devnull
            os.dup2(devnull.fileno(), 2)  # Redirect stderr to devnull
            
            # Params
            tcal_event = ROOT.TcalEvent()
            po_event = ROOT.TPOEvent()
            split = file.split("_")
            run_number, ievent = int(split[-2]), int(split[-1][:-5])
            event_mask = 0

            # Load event
            po_event_address = ROOT.AddressOf(po_event)
            result = tcal_event.Load_event(path, run_number, ievent, event_mask, po_event)

            # Reconstruct
            fPORecoEvent = ROOT.TPORecoEvent(tcal_event, tcal_event.fTPOEvent)
            fPORecoEvent.Reconstruct()
            fPORecoEvent.TrackReconstruct()
            fPORecoEvent.Reconstruct2DViewsPS()
            fPORecoEvent.ReconstructClusters(0)
            fPORecoEvent.Reconstruct3DPS()
            fPORecoEvent.PSVoxelParticleFilter()
            fPORecoEvent.ReconstructRearCals()
            fPORecoEvent.Fill2DViewsPS()

            xz_view = fPORecoEvent.Get2DViewXPS()
            yz_view = fPORecoEvent.Get2DViewYPS()
            z_view = fPORecoEvent.zviewPS
            rearcal_energydeposit = fPORecoEvent.rearCals.rearCalDeposit
            rearmucal_energydeposit = fPORecoEvent.rearCals.rearMuCalDeposit
            geom_detector = tcal_event.geom_detector

            # Retrieve true 3D hits and 2D projections
            true_hits = get_true_hits(tcal_event)
            if true_hits is None:
                print("Empty event {}".format(i))
                continue

            xz_proj = th2d_to_numpy(xz_view)
            yz_proj = th2d_to_numpy(yz_view)
            xy_projs = []
            for layer in range(20):
                view = z_view[layer]
                proj = th2d_to_numpy(view)
                xy_projs.append(proj)

            # Retrieve reco 3d hits
            reco_hits, reco_hits_true = get_reco_hits(fPORecoEvent, tcal_event, true_hits, geom_detector)

            # Global information
            run_number = po_event.run_number
            event_id = po_event.event_id        
            prim_vertex = po_event.prim_vx
            prim_vertex = np.array([prim_vertex.x(), prim_vertex.y(), prim_vertex.z()])
            iscc = bool(po_event.isCC)
            evis = po_event.Evis
            ptmiss = po_event.ptmiss
            in_neutrino = po_event.in_neutrino
            out_lepton = po_event.out_lepton
            in_neutrino_pdg = in_neutrino.m_pdg_id
            in_neutrino_momentum = np.array([in_neutrino.m_px, in_neutrino.m_py, in_neutrino.m_pz]) 
            in_neutrino_energy = in_neutrino.m_energy
            out_lepton_pdg = out_lepton.m_pdg_id
            out_lepton_momentum = np.array([out_lepton.m_px, out_lepton.m_py, out_lepton.m_pz]) 
            out_lepton_energy = out_lepton.m_energy

            # tracks
            tktracks = get_tracks(fPORecoEvent.fTKTracks)
            pstracks = get_tracks(fPORecoEvent.fPSTracks)

            np.savez_compressed('/scratch2/salonso/faser/events_v3_large/{}_{}'.format(run_number, event_id),
                                run_number = run_number,
                                event_id = event_id,
                                iscc = iscc,
                                evis = evis,
                                ptmiss = ptmiss,
                                rearcal_energydeposit = rearcal_energydeposit,
                                rearmucal_energydeposit = rearmucal_energydeposit,
                                prim_vertex = prim_vertex,
                                true_hits = true_hits,
                                reco_hits = reco_hits,
                                reco_hits_true = reco_hits_true,
                                xz_proj = xz_proj,
                                yz_proj = yz_proj,
                                xy_projs = xy_projs,
                                tktracks = tktracks,
                                pstracks = pstracks,
                                in_neutrino_pdg = in_neutrino_pdg,
                                in_neutrino_momentum = in_neutrino_momentum,
                                in_neutrino_energy = in_neutrino_energy,
                                out_lepton_pdg = out_lepton_pdg,
                                out_lepton_momentum = out_lepton_momentum,
                                out_lepton_energy = out_lepton_energy,
                               )

            # Restore stdout and stderr
            os.dup2(old_stdout_fno, 1)  # Restore the original stdout
            os.dup2(old_stderr_fno, 2)  # Restore the original stderr
            os.close(old_stdout_fno)
            os.close(old_stderr_fno)

# Main function to handle command-line arguments
if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Divide a list into chunks and select one chunk based on a number.")
    
    # Add arguments
    parser.add_argument('--number', type=int, required=True, help="Number (0-9) to select the chunk")
    parser.add_argument('--chunks', type=int, required=True, help="Number of chunks")
    parser.add_argument("--disable", action="store_true", default=False, help="Disable progressbar")

    # Parse arguments
    args = parser.parse_args()
    number = args.number
    chunks = args.chunks
    disable = args.disable

    # Validate number range
    if number < 0 or number > 9:
        raise ValueError("Number must be between 0 and 9")
  
    generate_events(number, chunks, disable)
    print("Done!")
 
