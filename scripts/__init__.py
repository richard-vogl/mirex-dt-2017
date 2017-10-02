# -*- coding: utf-8 -*-
"""
Created on Fri May 26 15:10:50 2017

@author: CarlSouthall
"""

from . import utils
import os


def DTCS(filenames, context, save_dir=None):
    location = utils.location_extract()
    Onsets = []
    Peaks = None
    for k in filenames:
        specs = utils.spec(k)
        AFs = utils.system_restore(specs, context, location)
        PP = utils.load_pp_param(context, location)
        Peaks = []
        for j in range(len(AFs)):
                Peaks.append(utils.meanPPmm(AFs[j][:,0],PP[int(context),j,0],PP[int(context),j,1],PP[int(context),j,2]))
        sorted_p = utils.sort_ascending(Peaks)

        out_filename = os.path.splitext(k)[0] + '.txt'
        if save_dir is not None:
            out_filename = os.path.join(save_dir, os.path.basename(out_filename))
        utils.print_to_file(sorted_p, out_filename)
#                     
    return Peaks
