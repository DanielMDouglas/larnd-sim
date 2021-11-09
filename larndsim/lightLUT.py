"""
Code adapted from https://github.com/PPKoller/ArgonCubeLUTSim/blob/main/module_0/lutSim.py
"""

import sys, time, math
import argparse
import numpy as np
from math import log, isnan
from . import consts    

def get_lut_geometry(lut_path):
    """
    Finds the max and minimum values of the x,y and z axis and the number of voxels in each direction.
    Args:
        lut_path (str): filename of numpy array (.npy) containing light calculation
    Returns:
        :obj:`numpy.ndarray`: 3x3 array of voxelization information (minimum, maximum, number of divisions) in each dimension
    """
    
    f = np.load(lut_path)
    lut_min = np.array([f['Min'][0],f['Min'][1],f['Min'][2]])
    lut_max = np.array([f['Max'][0],f['Max'][1],f['Max'][2]])
    lut_ndiv = np.array([f['NDivisions'][0],f['NDivisions'][1],f['NDivisions'][2]])

    return np.array([lut_min,lut_max,lut_ndiv])

def get_voxel(pos,lut_geometry):
    """
    Determines which voxel is being called based on the position of the edep. Voxels are indexed 0-2911
    Args:
        pos (:obj:`numpy.ndarray`): list of x, y, z coordinates within a generic TPC volume
        lut_geometry (obj:`numpy.ndarray`): 3x3 array of voxelization information 
            (minimum, maximum, number of divisions) in each dimension
    Returns:
        :obj:`numpy.float64`: index of the voxel containing the input position
    """
    
    (lut_min,lut_max,lut_ndiv) = lut_geometry
    vox_xyz = np.floor(pos/(lut_max-lut_min)*lut_ndiv).astype(int)+lut_ndiv/2
    voxel = vox_xyz[2]*lut_ndiv[0]*lut_ndiv[1]+vox_xyz[1]*lut_ndiv[0]+vox_xyz[0]

    return voxel


def get_half_det_copy(pos):
    """
    Determines in which TPC the edep takes place.
    Currently this is done based only on the x-position.
    This should actually call the detector properties yaml.
    Args:
        pos (:obj:`numpy.ndarray`): list of x, y, z coordinates within a detector geometry
    Returns:
        int: index of the TPC containing the input posision
    """
    
    tpc_x = math.floor(pos[0]/(consts.ModuleDimension[0]/2.))+consts.n_mod[0]/2*consts.n_tpc[0]

    return int(tpc_x)%2

def larnd_to_lut_coord(pos,lut_geometry):
    """
    Converts the LArND-sim coord. system to that of the LUT.
    LUT should be updated to LArND-sim system and this function removed.
    Args:
        pos (:obj:`numpy.ndarray`): list of x, y, z coordinates within a generic TPC
    Returns:
        :obj:`numpy.ndarray`: list of x, y, z coordinates translated to the LUT system
    """
    
    # access LUT geometry
    lut_min,lut_max,lut_ndiv = lut_geometry
    
    # should add another condition to be greater than whatever the thckness of the cathode is 
    lut_pos = pos + np.array([lut_min[0],220,0])

    return (lut_pos)

def calculate_light_incidence(t_data,lut_path,light_dat):
    """
    Simulates the number of photons read by each optical channel depending on where the edep occurs. 
    Also indicates the time it takes for a photon to reach the nearest photomultiplier tube (the "fastest" photon)
    Args:
        t_data (:obj:`numpy.ndarray`): track array containing edep segments, positions are used for lookup
        lut_path (str): filename of numpy array (.npy) containing light calculation
        light_dat (:obj:`numpy.ndarray`): to contain the result of light incidence calculation.
            this array has dimension (n_tracks, n_optical_channels) and each entry
            is a structure of type (n_photons_edep (float32), n_photons_det (float32), t0_det (float32))
            these correspond the number of photons produced by a given edep (stored only in the 0th channel),
            the number detected in each channel (n_photons_edep*visibility), and the time of earliest
            arrival at that channel.
    """
    
    # Loads in LUT root file
    np_lut = np.load(lut_path)
    
    # Obtains lut geometry
    lut_geometry = get_lut_geometry(lut_path)  
    
    # Data containers
    time = np.full((t_data['dE'].size,consts.n_op_channel*2),20.)
    tphotons = np.zeros((t_data['dE'].size,consts.n_op_channel*2))
    
    # Defines variables of global position
    x = t_data['x'] # The average of the position between x_start and x_end from the edep-sim file
    y = t_data['y']
    z = t_data['z']

    # Defining number of produced photons (from quencing.py)
    n_photons = light_dat['n_photons_edep'][:,0]

    # Loop edep positions
    for dE in range(len(t_data['dE'])):

        # Global position
        pos = (np.array((z[dE],y[dE],x[dE])))*consts.cm2mm

        sys.stdout.write('\r    current position: ' + str(pos) + ('(%.1f %%)' % ((dE+1)/float(len(t_data['dE']))*100)))
        sys.stdout.flush()
        
        # tpc
        tpc = get_half_det_copy(pos)
        
        # LUT position
        lut_pos = larnd_to_lut_coord(pos,lut_geometry) 

        # voxel containing LUT position
        voxel = get_voxel(lut_pos,lut_geometry)
        
        # Calls voxel data
        lut_vox = np_lut[np_lut['Voxel'] == voxel]
        
        op_dat = lut_vox['OpChannel']
        vis_dat = lut_vox['Visibility']
        T1_dat = lut_vox['T1']
        
        # loop voxel entry-list
        for entry in range(len(lut_vox)):

            op_channel = op_dat[entry]
            
            # Calculates the number of photons reaching each optical channel
            n_photons_read = vis_dat[entry]*n_photons[dE]
            
            # Flips op channels if edep occurs in tpc 2
            if (tpc==1):
                op_channel = (op_channel+consts.n_op_channel/2)%consts.n_op_channel
            
            # Determines the travel time of the "fastest" photon
            if (T1_dat[entry] < time[dE,int(op_channel)]):  
                time[dE,int(op_channel)] = T1_dat[entry]
            
            if (op_channel % 12) > 5: 
                n_photons_read *= consts.norm_lcm_acl
                
            # new index for storing light data the optical channels in the second TPC
            if tpc == 1:
                op_channel += 48
            
            # Assigns the number of photons read for each optical channel to an array
            tphotons[dE,int(op_channel)] += n_photons_read
    
    # Assigns data to the h5 file
    light_dat['n_photons_det'] += tphotons
    light_dat['t0_det'] += time
