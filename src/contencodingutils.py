import numpy as np
import pickle

def get_chipdata():
    """Get encoded_phis, time_stamps from Iris measurements.
    Format 
    [time unit (s), starting phase, final phase, output normalized powers
    for each measurement step (dim: 1x60) from one output (THIS IS THE ONE
    YOU ARE INTERESTED IN), non-normalized powers for all the chip modes
    (dim: 6x60), time where each measurement is taken]
    """
    with open(".\src\chipdata\nina_measurements0pi1", "r") as f:
    chip_data = pickle.load(f)
    return chip_data

def get_encoded_phis_time(chip_data):
    """ Use chip data to obtain inputs for get_detector_timebins function.
    #TODO: cut before and after output normalized powers
    for each measurement step are approximately constant. Then,
    stitch data together in one long np.array with ascending orders of starting and final phases.
    """
    
    phi_times = chip_data[:, [3,4]]

    return phi_times

def get_detector_timebins(phi_times, equi_time: bool = True, bin_number: int = 10):
    """ Use data, the encoded phases and corresponding time stamps,
        from photonic chip experiments to obtain time bins for the detector.
        Additionally this can be used to adapt the digital training of the 
        chip to the curve of the continuous variable encoding 'sweep'.
        Arguments:
            phi_times: contains encoded phases with time stamps depending on detector time resolution.
            equi_time: either fit parameters of chip digitally such that equi-sized time bins
                       are used (if set to True). Otherwise, equi-distant bins in the encoded phases 
                       are used. TODO: determine experimentally which one works better (guess it will be equi-time.)
            bin_number: number of classical data points to be encoded continuously.
        Return: 
            time_stamps: time stamps for when detector should allocate measurements to next data point.
    """

    if equi_time == False:
        x_max = phi_times[:,1].max()
        x_min = phi_times[:,1].min()
        xbins = [x_min+m*(x_max-x_min)/bin_number for m in range(bin_number+1)]
        for m in range(bin_number+1):
            # find closest time stamp to xbins[m]
            # add time stamp to list
            time_stamps = []
    else: 
        # here we have equi-time bins, but then the encoded phase bins have different sizes
        # get time stamp zero for x_min
        t_min = phi_times[:,2].min()
        t_max = phi_times[:,2].max() # this should be the time point when max phi_enc becomes approximately constant
        time_stamps = [t_min+m*(t_max-t_min)/bin_number for m in range(bin_number+1)]
    
    return time_stamps