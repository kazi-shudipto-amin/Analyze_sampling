import numpy as np
import math
import mdtraj as md
from matplotlib import pyplot as pp
import pandas as pd

def structural_histogram(trajectory, d_c):
    """
    Generates a structural histogram of states based on a cutoff RMSD;
    Inputs- trajectory: trajectory of interest, d_c: a cutoff RMSD;
    Returns- 1) A dictionary of clusters; the keys are the randomly chosen refernce frames, 
              and the values are the frames within the cutoff rmsd from the reference
             2) a structural histogram
             3) a pandas dataframe- columns represent each reference frame;
                                    rows represent rmsd between each frame 
                                                   and the reference frame
    """
    
    traj = md.load(trajectory)
    N = traj.n_frames
    available_frames = np.arange(0, N, 1)
    
    ### CLUSTERING STRUCTURES
    clusters = {}
    reference_structures = []
    df_matrix = pd.DataFrame()
    
    while len(available_frames) > 0:
        """
        Once a reference structure is selected, the reference structure
        and the frames within the cutoff are removed from the trajectory
        (frames_to_exclude: a list of frames to be removed from the traj)
        """
        frames_to_exclude = []
        
        # randomly selecting a reference frame
        sample = np.random.choice(available_frames, replace=True, p=None)
        frames_to_exclude.append(sample)
        
        # appending the index of the reference frame as a key for binning
        clusters.update( {'%s'%sample : [sample]})
    
        ref_struct = traj[sample]
        reference_structures.append(ref_struct)
    
        # align trajectory to the reference frame and caculate rmsd
        aligned_traj = traj.superpose(traj, frame=sample, atom_indices=None, 
                             ref_atom_indices=None, parallel=True)
    
        rmsd_from_reference = md.rmsd(aligned_traj, aligned_traj, 
                                     frame=sample, atom_indices=None, 
                                     parallel=True, precentered=False)
        
        # Saving the reference frame along with the rmsd to each frame in the trajectory
        # in a pandas dataframe
        df_matrix["%s"%sample] = rmsd_from_reference
        
        # Index: an array of indices sorted according to the rsmd_from_frames;
        # removing indices matching the frames that available to choose from
        index = np.argsort(rmsd_from_reference)
        index = [item for item in index if item in available_frames]
        
        rmsds = []
        for i in index:
            rmsds.append(rmsd_from_reference[i])
        
        l = len(rmsds[1:])

        for i in range(l):
            rmsd = rmsds[i+1]
            if rmsd < d_c:
                frame_number = index[i+1]
                frames_to_exclude.append(frame_number)
                clusters['%s'%sample].append(frame_number)

        available_frames = [item for item in available_frames if item not in frames_to_exclude]
    
    ### Classification
    df_matrix['Belongs_to'] = df_matrix.idxmin(axis=1)
    population = df_matrix['Belongs_to'].value_counts()
    num_rows = df_matrix.shape[0]
    norm_population = population.divide(other=num_rows)
    
    # Plotting bar diagram
    fig, ax = pp.subplots(nrows=1, ncols=1, facecolor=('#e9f1f3'),
                          figsize=(12,7))

    norm_population.plot.bar(color='g', alpha=0.3, edgecolor='black', ax=ax)


    
    return fig, clusters, df_matrix