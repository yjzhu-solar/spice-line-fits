import copy, os, pickle, resource, numpy as np
from scipy.io import readsav
import sys

def masked_median_filter(data,mask,radius,footprint=None,missing=0.0,footprint_ind_offset=0):
    
    if(footprint is None):
        if(np.isscalar(radius)): radarr = np.array([radius]*data.ndim)
        else: radarr = radius
        coordas = np.indices(2*radarr+1)
        for i in range(0,len(radarr)): coordas[i] = (coordas[i]-radarr[i])/radarr[i]
        radii = np.sum(coordas**2,axis=0)**0.5
        footprint = radii <= 1
        
    flatinds = np.ravel_multi_index(np.indices(data.shape),data.shape).flatten()
    footprint_inds = np.ravel_multi_index(np.indices(footprint.shape),footprint.shape)
    footprint_inds = np.array(np.unravel_index(footprint_inds[footprint],footprint.shape))
    footprint_inds = footprint_inds.transpose(np.roll(np.arange(footprint_inds.ndim),-1))
    data_filt = np.zeros(data.size)+missing
    footprint_pad = np.floor(0.5*np.array(footprint.shape)).astype(np.int32)
    footprint_pad = np.array([footprint_pad,footprint_pad]).T
   
    data_fppad = np.pad(data,footprint_pad)
    dat_pad_shape = data_fppad.shape
    data_fppad = data_fppad.flatten()
    mask_fppad = np.pad(mask,footprint_pad).flatten()
    tparg = np.roll(np.arange(footprint_inds.ndim),1)
    data_filt = _masked_medfilt_inner(flatinds,data,footprint_inds,footprint_ind_offset,tparg,dat_pad_shape,data_fppad,mask_fppad,data_filt)

    return data_filt.reshape(data.shape)

def _masked_medfilt_inner(flatinds,data,footprint_inds,footprint_ind_offset,tparg,dat_pad_shape,data_fppad,mask_fppad,data_filt):
    for ind in flatinds:
        ijkpad = np.unravel_index(ind,data.shape)+footprint_inds-footprint_ind_offset
        ijkpad = np.ravel_multi_index(ijkpad.transpose(tparg),dat_pad_shape)
        dat = data_fppad[ijkpad]
        good = mask_fppad[ijkpad]
        if(np.sum(good) > 0): data_filt[ind] = np.median(dat[good])
    return data_filt

def bindown(d,n):
    inds = np.ravel_multi_index(np.floor((np.indices(d.shape).T*n/np.array(d.shape))).T.astype(np.uint32),n)
    return np.bincount(inds.flatten(),weights=d.flatten(),minlength=np.prod(n)).reshape(n)

def bindown2(d,f):
    n = np.round(np.array(d.shape)/f).astype(np.int32)
    inds = np.ravel_multi_index(np.floor((np.indices(d.shape).T*n/np.array(d.shape))).T.astype(np.uint32),n)
    return np.bincount(inds.flatten(),weights=d.flatten(),minlength=np.prod(n)).reshape(n)
    
def binup(d,f):
    n = np.round(np.array(d.shape)*np.round(f)).astype(np.int32)
    inds = np.ravel_multi_index(np.floor((np.indices(n).T/np.array(f))).T.astype(np.uint32),d.shape)
    return np.reshape(d.flatten()[inds],n)
  
def get_mask_errs(dat_cube, iris_err_fac, error_cube=None, filt_thold=2.5):
    dat_arr = copy.deepcopy(dat_cube)
    dat_filt = masked_median_filter(dat_arr,(np.isnan(dat_arr)==0),np.array([1,2,1]))
    if(error_cube is None): error_cube = ((iris_err_fac**2+np.clip(dat_arr,0,None)*iris_err_fac)**0.5).astype('float64')

    dat_median = np.nanmedian(np.abs(dat_filt))
    dat_mask = (np.isnan(dat_arr) + (dat_arr < -3.00*error_cube) + np.isnan(error_cube) +
                  (np.abs(dat_arr-dat_filt) > filt_thold*(dat_median+np.abs(dat_filt)))) > 0

    return dat_mask, error_cube

def get_spice_err(data, header, detector=None, npx=1, verbose=True):
	# Error conversion factors for short and longwave detectors, respectively, from 
	# Appendix B of Huang et al, A&A 673, A82.
	#    F  -- Intensifier noise factor,
	#    G  -- Camera conversion gain in DN/photon
	#    RN -- Read noise sigma in DN
	#    DC -- Dark current in DN/s
	F  = [1,1.6]
	G  = [3.58,0.57]
	RN = [6.9,6.9] 
	DC = [0.89,0.54]
	exptime = header['XPOSURE']
	if(detector is None): detector = header['DETECTOR']
	i = 1*(detector != 'SW')
	if(header['LEVEL']=='L2'): radcal=header.get('RADCAL',header.get('RCALAVG'))
	else: radcal=1
	if(verbose): print('Estimating error for '+detector+' detector, index '+str(i))
	return np.sqrt(radcal*np.abs(data)*G[i]*F[i]**2+npx*RN[i]**2+npx*DC[i]*exptime)/radcal

