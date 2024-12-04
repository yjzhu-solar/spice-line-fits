import copy, numpy as np
from astropy.nddata import NDData
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator as lndi

default_bin_fac = [3,3]

# Placing the binup and bindown routines here so that the skew correction is
# self contained in this one file. They're also present in util, though:
def bindown2(d,f):
    n = np.round(np.array(d.shape)/f).astype(np.int32)
    inds = np.ravel_multi_index(np.floor((np.indices(d.shape).T*n/np.array(d.shape))).T.astype(np.uint32),n)
    return np.bincount(inds.flatten(),weights=d.flatten(),minlength=np.prod(n)).reshape(n)
    
def binup(d,f):
    n = np.round(np.array(d.shape)*np.round(f)).astype(np.int32)
    inds = np.ravel_multi_index(np.floor((np.indices(n).T/np.array(f))).T.astype(np.uint32),d.shape)
    return np.reshape(d.flatten()[inds],n)
  
# This code applies a 'skew' correction in order to remove x-y-lambda PSF artifacts seen
# in SPICE data. These artifacts make features appear to shift across the solar surface
# if a image sequence scanning through wavelength is plotted.  When spectral lines are 
# fitted to affected data, the result is that apparent blue and red doppler shift features
# appear at the borders of bright regions in the data, when none are present.
# This compromises the scientific utility of SPICE Dopplers, and a reverse shift is therefore
# applied to the data to counteract it. However, the artifact is not truly a shift but is
# instead a rotation of an elliptical PSF in x, y and lambda. Using the reverse shift/skew
# as a correction for this PSF removes the blue and red doppler shift features. However, it 
# also ends up moving features in x and y, so a further distortion removing step to be applied
# after fitting. This is done by the routines deskew_doppler and deskew_fits below.
# Inputs to skew_correct:
#   dat: the data to be corrected, of dimensions [nx,ny,nlambda]
#   xlshift: the x-lambda shift to be applied, with units of (for example) arcseconds/angstrom
#   ylshift: the y-lambda shift to be applied, with units of (for example) arcseconds/angstrom
#   dx: The spatial extent of the pixels in x (e.g., arcseconds). Should match those in xlshift
#   dy: The spatial extent of the pixels in y (e.g., arcseconds). Should match those in ylshift
#   dl: The wavelength extent of the pixels in lambda (e.g., angstroms). Should match xlshift & ylshift
#   l0: The wavelength of the first wavelength pixel.
#   skew_bin_facs: In order to reduce interpolation artifacts, the images are binned up by these factors in x & y.
#   lcen: Reference wavelength to shift around. Ideally should be the average line center wavelength
#   lambdas: Optional input array of wavelengths
def skew_correct(dat, xlshift, ylshift, dx, dy, dl, l0, skew_bin_facs=default_bin_fac, lcen=None, lambdas=None):

	nl = dat.shape[2]
	if(lambdas is None): lambdas = l0+dl*np.arange(nl)
	if(lcen is None): lcen = np.mean(lambdas)

	# Set up the coordinate arrays used by the interpolator:
	spice_x = dx*np.arange(dat.shape[0]*skew_bin_facs[0])/skew_bin_facs[0]
	spice_y = dy*np.arange(dat.shape[1]*skew_bin_facs[1])/skew_bin_facs[1]
	[spice_xa0,spice_ya0] = np.indices(np.array(skew_bin_facs,dtype=np.int32)*dat.shape[0:2])
	spice_xa0 = dx*spice_xa0/skew_bin_facs[0]
	spice_ya0 = dy*spice_ya0/skew_bin_facs[1]
	spice_xya0 = np.array([spice_xa0.flatten(),spice_ya0.flatten()]).transpose()
	
	# Loop over wavelengths:
	spicedat_skew = np.zeros(dat.shape)
	for i in range(0,nl):
		# Bin up data and set up interpolator for this wavelength:
		rgi = RegularGridInterpolator((spice_x,spice_y),binup(dat[:,:,i],skew_bin_facs),bounds_error=False,fill_value=None)

		xshift = spice_xa0 - xlshift*(lambdas[i]-lcen) # How much to shift in x for this wavelength
		yshift = spice_ya0 - ylshift*(lambdas[i]-lcen) # How much to shift in y for this wavelength
		spice_xya = np.array([xshift,yshift]).transpose([1,2,0])
		
		# Interpolate and bin data back down for this wavelength:
		spicedat_skew[:,:,i] = bindown2(rgi(spice_xya),skew_bin_facs)/np.prod(skew_bin_facs)

	return spicedat_skew # Done.


def deskew_linefit_window(win_in, xlshift, ylshift, binfacs=default_bin_fac, lcen=None, lambdas=None, dopp_key='centers'):
	window = copy.deepcopy(win_in)
	for linkey in win_in:
		line = win_in[linkey]
		doppler = line.get(dopp_key)
		for parmkey in line:
			parm = line[parmkey]; hdr = parm.meta
			hdr_wavcen = hdr['CRVAL3'] + (0.5*hdr['NAXIS3']-(hdr['CRPIX3']-0.5))*hdr['CDELT3']
			shape = np.array([hdr['NAXIS1'],hdr['NAXIS2']],dtype=np.int32)
			if(lcen is None): lcen = 10*hdr_wavcen
			if(doppler is None): doppler = NDData(10*hdr_wavcen + np.zeros(shape))
			dx, dy = hdr['CDELT1'],hdr['CDELT2']

			# Set up the coordinate arrays used by the interpolator:
			spice_x = dx*np.arange(shape[0]*binfacs[0])/binfacs[0]
			spice_y = dy*np.arange(shape[1]*binfacs[1])/binfacs[1]
			[spice_xa0,spice_ya0] = np.indices((binfacs*shape).astype(np.int32))
			spice_xa0 = dx*spice_xa0/binfacs[0]
			spice_ya0 = dy*spice_ya0/binfacs[1]
			
			# Compute how much each line profile/pixel in the Doppler image was x-y shifted,
			# with the same the up-sampling as in skew_correct:
			dlambdas = binup(doppler.data.squeeze()-lcen,binfacs)
			xshift = spice_xa0 - xlshift*dlambdas
			yshift = spice_ya0 - ylshift*dlambdas

			for i in range(0,parm.data.shape[2]):
				for j in range(0,parm.data.shape[3]):
					datagood = binup(parm.uncertainty.array[:,:,i,j] > 0,binfacs)
					dat = binup(parm.data[:,:,i,j],binfacs)[datagood].flatten()
					err = binup(parm.uncertainty.array[:,:,i,j],binfacs)[datagood].flatten()
					spice_xya = np.array([xshift[datagood].flatten(),yshift[datagood].flatten()]).T
					dat_lndi = bindown2(lndi(spice_xya, dat)(spice_xa0,spice_ya0),binfacs)/np.prod(binfacs)
					err_lndi = bindown2(lndi(spice_xya, err)(spice_xa0,spice_ya0),binfacs)/np.prod(binfacs)
					err_lndi[np.isnan(dat_lndi)]=parm.meta['BAD_ERR']; dat_lndi[np.isnan(dat_lndi)]=0
					window[linkey][parmkey].uncertainty.array[:,:,i,j] = err_lndi
					window[linkey][parmkey].data[:,:,i,j] = dat_lndi

	return window
