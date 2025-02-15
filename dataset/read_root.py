#!pip install uproot
from ROOT import TFile, TTree, std
import ROOT
import glob
import numpy as np
import tqdm
import argparse

path = "/scratch2/salonso/faser/FASERCALDATA_v3.5/"
path_true_template = '/scratch2/salonso/faser/FASERCALDATA_v3.5/FASERG4-Tcalevent_{}_{}.root'
paths_true = glob.glob("/scratch2/salonso/faser/FASERCALDATA_v3.5/*")
paths_reco = glob.glob("/scratch2/salonso/faser/FASERCALRECODATA_v3.5/*")
print(len(paths_true), len(paths_reco))
print(paths_true[0], paths_reco[0])

#from ROOT import TFile
#ROOT.gSystem.Load("/scratch3/Simulation/PLATONOS/buildSaulLib/libOS.so")
#ROOT.gSystem.Load("/scratch5/FASER/V3.1_test/FASER/Batch/libTPORec.so")
ROOT.gSystem.Load("/scratch5/FASER/V3.1/FASER/Python_io/lib/ClassesDict.so")

# Placeholder for class objects
tcal_event = ROOT.TcalEvent()
tporeco_event = ROOT.TPORecoEvent()

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

def generate_events(number, chunks, disable):
    chunks = divide_list_into_chunks(paths_reco, num_chunks=chunks)
    chunk = chunks[number]
   
    print("First path: {}".format(chunk[0]))
    print("Number of files: {}/{}".format(len(chunk), len(paths_reco)))

    # Iterate over reconstruction files
    t = tqdm.tqdm(enumerate(chunk), total=len(chunk), disable=disable)
    for i, filename_reco in t:
        f_reco = TFile(filename_reco, "read")
        #f_reco.ls()
        reco_event = f_reco["RecoEvent"]
        #reco_event.Print()
        
        entries = reco_event.GetEntries()
        #print(entries)
        
        reco_event.SetBranchAddress("TPORecoEvent", tporeco_event)
    
        for entry in range(entries):
            reco_event.GetEntry(entry)
    
            # Global info
            po_event = tporeco_event.GetPOEvent()
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
            jet_momentum = np.array([po_event.jetpx, po_event.jetpy, po_event.jetpz])

            # Tracks
            tktracks = get_tracks(tporeco_event.fTKTracks)
            pstracks = get_tracks(tporeco_event.fPSTracks)
    
            # Views
            xz_view = tporeco_event.Get2DViewXPS()
            yz_view = tporeco_event.Get2DViewYPS()
            z_view = tporeco_event.zviewPS
            
            # rear calorimeter, mutag and fasercal deposited energies
            rearcal_energydeposit = tporeco_event.rearCals.rearCalDeposit
            rearhcal_energydeposit = tporeco_event.rearCals.rearHCalDeposit
            rearmucal_energydeposit = tporeco_event.rearCals.rearMuCalDeposit
            rearhcalmodules = np.zeros(shape=(9,))
            fasercalmodules = np.zeros(shape=(15,))
            for x in tporeco_event.rearCals.rearHCalModule:
                rearhcalmodules[x.moduleID] = x.energyDeposit
            for x in tporeco_event.faserCals:
                fasercalmodules[x.ModuleID] = x.EDeposit
            fasercal_energydeposit = fasercalmodules.sum()

            # geometry
            geom_detector = tporeco_event.geom_detector
    
            # Retrieve the corresponding true event
            po_event_address = ROOT.AddressOf(po_event)
            event_mask = 0
            result = tcal_event.Load_event(path, run_number, event_id, event_mask, po_event)
    
            # True and reco hits
            true_hits = get_true_hits(tcal_event)
            reco_hits, reco_hits_true = get_reco_hits(tporeco_event, tcal_event, true_hits, geom_detector)
    
            np.savez_compressed('/scratch2/salonso/faser/events_v3.5/{}_{}'.format(run_number, event_id),
                                run_number = run_number,
                                event_id = event_id,
                                iscc = iscc,
                                evis = evis,
                                ptmiss = ptmiss,
                                rearcal_energydeposit = rearcal_energydeposit,
                                rearhcal_energydeposit = rearhcal_energydeposit,
                                rearmucal_energydeposit = rearmucal_energydeposit,
                                fasercal_energydeposit = fasercal_energydeposit,
                                rearhcalmodules = rearhcalmodules,
                                fasercalmodules = fasercalmodules,
                                prim_vertex = prim_vertex,
                                true_hits = true_hits,
                                reco_hits = reco_hits,
                                reco_hits_true = np.array(reco_hits_true, dtype=object),
                                #xz_proj = xz_proj,
                                #yz_proj = yz_proj,
                                #xy_projs = np.array(xy_projs, dtype=object),
                                tktracks = np.array(tktracks, dtype=object),
                                pstracks = pstracks,
                                in_neutrino_pdg = in_neutrino_pdg,
                                in_neutrino_momentum = in_neutrino_momentum,
                                in_neutrino_energy = in_neutrino_energy,
                                out_lepton_pdg = out_lepton_pdg,
                                out_lepton_momentum = out_lepton_momentum,
                                out_lepton_energy = out_lepton_energy,
                                jet_momentum = jet_momentum,
                               )
             
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
    if number < 0 or number >= chunks:
        raise ValueError("Number must be between 0 and 9")
  
    generate_events(number, chunks, disable)
    print("{}/{} Done!".format(number, chunks)
)
