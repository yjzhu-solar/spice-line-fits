# Utility modules for SPICE line fitting and skew-based correction to PSF
# Doppler artifacts.
import copy, os, pickle, resource, numpy as np
from scipy.io import readsav
import sys

# Perform a median filter on n-dimensional data with an elliptical footprint of axis radii radius
# (or using optional input keyword 'footprint'), while masking out specified pixels/elements of the array
# (omitting them from the medians).
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

# Quick routines to do integer up and down binning on n-dimensional numpy arrays:
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

# This routine computes a bad pixel mask for SPICE data. The parameter err_fac is retained
# for legacy purposes and is best not used. Instead, compute an error cube using get_spice_err
# or official routines from the SPICE instrument software. If the impulse_filter argument is true,
# the routine will also mask out pixels based on a median filter. filt_thold is the threshold for filtering
# and the footprint of the median filter is a 3D ellipse with axis radii given by filter_footprint.
# Defaults for these are False, 2.5, and [1,2,1], respectively. Note that this filtering is slow.
def get_mask_errs(dat_cube, err_fac, error_cube=None, impulse_filter=False, filt_thold=2.5, filter_footprint=[1,2,1]):
	dat_arr = copy.deepcopy(dat_cube)
	if(impulse_filter): dat_filt = masked_median_filter(dat_arr,(np.isnan(dat_arr)==0),np.array(filter_footprint))
	if(error_cube is None): error_cube = ((err_fac**2+np.clip(dat_arr,0,None)*err_fac)**0.5).astype('float64')

	if(impulse_filter):
		dat_median = np.nanmedian(np.abs(dat_filt))
		dat_mask = (np.isnan(dat_arr) + (dat_arr < -3.00*error_cube) + np.isnan(error_cube) +
					(np.abs(dat_arr-dat_filt) > filt_thold*(dat_median+np.abs(dat_filt)))) > 0
	else:
		dat_mask = (np.isnan(dat_arr) + (dat_arr < -3.00*error_cube) + np.isnan(error_cube))

	return dat_mask, error_cube

# This routine estimates the range in y of the main SPICE FOV (i.e., excluding the dumbbells
# and the dark bands around the dumbbells.
# band_separation is separation of lower edges of dark region between dumbbells and main FOV,
# band_size is height of those dark regions, band1_min is minimum y to check for lower edge
# of lower dark band (top of lower dumbbell), window_safe_radius is distance from center
# (assumed to be halfway between lower and upper dark band) to return as the SPICE 'regular'
# (non-masked, non-dumbbell) data y range.
def get_spice_data_yrange(spice_dat_in, window_safe_radius=288, band_separation=655, band_size=45, 
						  band1_min=None, band1_max=None):
	if(band1_min is None): band1_min = 0
	if(band1_max is None): band1_max = spice_dat_in.shape[1]-band_separation-band_size
	spice_dat = copy.deepcopy(spice_dat_in) - np.nanmean(np.nanmin(spice_dat_in,axis=0))
	signal_cube = spice_dat
	signal = np.nanmean(signal_cube,axis=(0,2))
	signal_windowed = []
	for i in range(band1_min,band1_max):
		signal_windowed.append(np.nanmean(np.hstack([signal[i:i+band_size],signal[(i+band_separation):(i+band_size+band_separation)]])))
	
	band1_bestmin = np.argmin(signal_windowed)+band1_min
	band1_bestmax = band1_bestmin+band_size
	band2_bestmin, band2_bestmax = band1_bestmin+band_separation, band1_bestmax+band_separation
	data_center_px = round(band1_bestmin+0.5*band_size+0.5*band_separation)
	reg_data_min, reg_data_max = data_center_px-window_safe_radius, data_center_px+window_safe_radius
	
	return reg_data_min, reg_data_max, band1_bestmin, band1_bestmax, band2_bestmin, band2_bestmax, [np.arange(band1_min,band1_max),signal_windowed], signal_cube

# Make a figure to check if the main FOV y range estimation is correct:
def make_yrange_check_plot(spice_dat, spice_mask, spice_hdr, ymin, ymax, band1_ymin, band1_ymax, 
							band2_ymin, band2_ymax, snr_windowed, snr_cube, plot_dir=None):
	import matplotlib.pyplot as plt
	spice_err = get_spice_err(spice_dat, spice_hdr, verbose=False)
	snr = np.nansum(spice_dat**2/spice_err**2,axis=2)
	fig,axes = plt.subplots(figsize=[19,9.5],nrows=1,ncols=3)
	plt.suptitle(spice_hdr['DATE-OBS']+' data region estimate for '+spice_hdr['EXTNAME'])
	axes[0].plot(snr_windowed[0], snr_windowed[1]/np.max(snr_windowed[1]))
	axes[0].plot([0,len(snr_windowed[0])-1],[np.min(snr_windowed[1])/np.max(snr_windowed[1]),np.min(snr_windowed[1])/np.max(snr_windowed[1])],
				 [band1_ymin,band1_ymin],[0,1.25],color='green',linestyle='dashed')
	axes[0].set(title='Signal vs guessed dark band position',xlabel='Offset of bottom of lower band',
				ylabel='Signal (normalized to one)',ylim=[-0.5,1.25])
	axes[0].legend(['Signal','Position estimate (min Signal)'])
	axes[1].set(title='Signal and dark band best fit')
	signal_img = np.clip(np.nanmean(snr_cube,axis=2),0,None)
	signal_img_max = 1.5*np.sort(signal_img[np.isfinite(signal_img)])[round(0.975*np.sum(np.isfinite(signal_img)))]
	axes[1].imshow(signal_img.T**0.5,aspect=spice_hdr['CDELT2']/spice_hdr['CDELT1'],vmax=signal_img_max**0.5)
	axes[1].plot([0,spice_dat.shape[0]-1],[ymin,ymin],color='magenta',linewidth=3)
	axes[1].plot([0,spice_dat.shape[0]-1],[band1_ymin,band1_ymin],
				 [0,spice_dat.shape[0]-1],[band1_ymax,band1_ymax],
				 [0,spice_dat.shape[0]-1],[band2_ymin,band2_ymin],
				 [0,spice_dat.shape[0]-1],[band2_ymax,band2_ymax],
				 color='red',linestyle='dashed',linewidth=3)
	axes[1].plot([0,spice_dat.shape[0]-1],[ymax,ymax],color='magenta',linewidth=3)
	axes[1].legend(['Main raster position estimate','Dark Band Region Fit'],loc='center')

	asp1 = np.diff(axes[1].get_xlim())[0] / np.diff(axes[1].get_ylim())[0]
	asp0 = np.diff(axes[0].get_xlim())[0] / np.diff(axes[0].get_ylim())[0]
	axes[0].set_aspect(spice_hdr['CDELT2']/spice_hdr['CDELT1']*asp0/asp1)
	
	axes[2].set(title='Fittable pixel mask')
	axes[2].imshow(spice_mask.T,aspect=spice_hdr['CDELT2']/spice_hdr['CDELT1'])
	axes[2].plot([0,spice_dat.shape[0]-1],[ymin,ymin],color='magenta',linewidth=3)
	axes[2].plot([0,spice_dat.shape[0]-1],[band1_ymin,band1_ymin],
				 [0,spice_dat.shape[0]-1],[band1_ymax,band1_ymax],
				 [0,spice_dat.shape[0]-1],[band2_ymin,band2_ymin],
				 [0,spice_dat.shape[0]-1],[band2_ymax,band2_ymax],
				 color='red',linestyle='dashed',linewidth=3)
	axes[2].plot([0,spice_dat.shape[0]-1],[ymax,ymax],color='magenta',linewidth=3)


	labelstr = spice_hdr['DATE-OBS']+' '+spice_hdr['EXTNAME']
	labelstr = labelstr.replace('-','').replace(':','').replace('  ','_').replace(' ','_')
	labelstr = labelstr.replace('/','_')
	if(plot_dir is None): plot_dir = '../shift_vars/figs/ypos_estimate/'
	plt.savefig(os.path.join(plot_dir,'raster_pos_'+labelstr+'.png'))
	plt.close()

# Estimate read and shot noise for a SPICE observation:
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


