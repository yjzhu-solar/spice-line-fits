# The codes in this paper (primarily skew_correct) perform the skew-based PSF correction.
# See comments on skew_correct, and skew_parameter_search for how to estimate the best
# skew parameters.

import copy, numpy as np, astropy.units as u
from astropy.nddata import NDData
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator as lndi
from linefit_leastsquares import lsq_fitter, check_for_waves
from util import get_mask_errs, get_spice_err, get_spice_data_yrange, make_yrange_check_plot
from linefit_storage import linefits

default_bin_fac = [3,3] # How much to bin up when interpolating. Large values result in slightly higher quality but take longer.

# Placing the binup and bindown routines here so that the skew correction is
# self contained in this one file. They're also present in util:
def bindown2(d,f):
    n = np.round(np.array(d.shape)/f).astype(np.int32)
    inds = np.ravel_multi_index(np.floor((np.indices(d.shape).T*n/np.array(d.shape))).T.astype(np.uint32),n)
    return np.bincount(inds.flatten(),weights=d.flatten(),minlength=np.prod(n)).reshape(n)
    
def binup(d,f):
    n = np.round(np.array(d.shape)*np.round(f)).astype(np.int32)
    inds = np.ravel_multi_index(np.floor((np.indices(n).T/np.array(f))).T.astype(np.uint32),d.shape)
    return np.reshape(d.flatten()[inds],n)
  
# Names and wavelengths of the standard lines to search:
default_linelist = {'Ar VIII+S III 700':700.3, 'O III 703':702.9, 'O III 704':703.9, 'Mg IX 706':706.0,
			'O II 718':718.5, 'S IV 745':744.9, 'S IV 748':748.4, 'S IV 750':750.2,
			'O V 759':758.7, 'S IV+O V 759':759.4, 'O V 760':760.3, 'O V 762':762.0,
			'N IV 765':765.1, 'Ne VIII 770':770.4, 'Mg VIII 772':772.3, 'Ne VIII 780':780.3,
			'S V 786':786.5, 'O IV 787':787.7, 'O IV 790':790.1, 'Ly Gamma 972':972.5,
			'C III 977':977.0, 'O I +- Na VI 989':988.7, 'N III 990':989.8, 'N III 992':991.6,
			'H I (+ O I) 1025':1025.7, 'O I 1027':1027.4, 'O VI 1032':1031.9, 'C II 1036':1036.5,
			'O VI 1037':1037.6}

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
#   dat: The data to be corrected, of dimensions [nx,ny,nlambda]
#   hdr: Header for the data to be corrected. Assumes axis 1 is x (raster), 2 is y (slit), 3 is wavelength.
#   xlshift: the x-lambda shift to be applied. Units: header CDELT1 unit per wav_unit (e.g., arcsec/angstrom)
#   ylshift: the y-lambda shift to be applied. Units: header CDELT2 unit per wav_unit (e.g., arcsec/angstrom)
#   skew_bin_facs: In order to reduce interpolation artifacts, the images are binned up by these factors in x & y.
#   lcen: Reference wavelength to shift around. Ideally should be the average line center wavelength.
#         Default is average of wavelengths in window.
#   lambdas: Optional input array of wavelengths
#   wav_unit: Wavelength units assumed in astropy units -- default: Angstrom
def skew_correct(dat, hdr, xlshift, ylshift, skew_bin_facs=default_bin_fac, 
				lcen=None, lambdas=None, offsets=[[0,0]], wav_unit=u.Angstrom):

	nl = dat.shape[2]
	wav_fac = 1.0/wav_unit.to(hdr['CUNIT3']) # Convert header wavelength unit to units for shift
	dx, dy, dl = hdr['CDELT1'], hdr['CDELT2'], wav_fac*hdr['CDELT3']
	l0 = wav_fac*hdr['CRVAL3']-dl*hdr['CRPIX3']
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

		for offset in offsets:
			xshift = spice_xa0 - xlshift*(lambdas[i]-lcen) - offset[0] # How much to shift in x for this wavelength
			yshift = spice_ya0 - ylshift*(lambdas[i]-lcen) - offset[1] # How much to shift in y for this wavelength
			spice_xya = np.array([xshift,yshift]).transpose([1,2,0])
			
			# Interpolate and bin data back down for this wavelength:
			spicedat_skew[:,:,i] += bindown2(rgi(spice_xya),skew_bin_facs)/np.prod(skew_bin_facs)/len(offsets)

	return spicedat_skew # Done.


# The skew correction introduces subpixel shifts in x and y depending on the wavelength at each pixel.
# this algorithm removes them with a non-uniform-gridded interpolator. The line fit results need to be
# in the containers provided by linefit_storage.py
def deskew_linefit_window(win_in, xlshift, ylshift, skew_bin_facs=default_bin_fac, lcen=None, lambdas=None, 
			dopp_key='centers', dokeys=None):
	window = copy.deepcopy(win_in)
	for linkey in win_in:
		line = win_in[linkey]
		doppler = line.get(dopp_key)
		if(dokeys is None): keys = line.keys()
		else: keys = dokeys
		for parmkey in keys:
			parm = line[parmkey]; hdr = parm.meta
			hdr_wavcen = hdr['CRVAL3'] + (0.5*hdr['NAXIS3']-(hdr['CRPIX3']-0.5))*hdr['CDELT3']
			shape = np.array([hdr['NAXIS1'],hdr['NAXIS2']],dtype=np.int32)
			if(lcen is None): lcen = 10*hdr_wavcen
			if(doppler is None): doppler = NDData(10*hdr_wavcen + np.zeros(shape))
			dx, dy = hdr['CDELT1'],hdr['CDELT2']

			# Set up the coordinate arrays used by the interpolator:
			spice_x = dx*np.arange(shape[0]*skew_bin_facs[0])/skew_bin_facs[0]
			spice_y = dy*np.arange(shape[1]*skew_bin_facs[1])/skew_bin_facs[1]
			[spice_xa0,spice_ya0] = np.indices((skew_bin_facs*shape).astype(np.int32))
			spice_xa0 = dx*spice_xa0/skew_bin_facs[0]
			spice_ya0 = dy*spice_ya0/skew_bin_facs[1]
			
			# Compute how much each line profile/pixel in the Doppler image was x-y shifted,
			# with the same the up-sampling as in skew_correct:
			dlambdas = binup(doppler.data.squeeze()-lcen,skew_bin_facs)
			xshift = spice_xa0 - xlshift*dlambdas
			yshift = spice_ya0 - ylshift*dlambdas

			for i in range(0,parm.data.shape[2]):
				for j in range(0,parm.data.shape[3]):
					datagood = binup(parm.uncertainty.array[:,:,i,j] > 0,skew_bin_facs)
					dat = binup(parm.data[:,:,i,j],skew_bin_facs)[datagood].flatten()
					err = binup(parm.uncertainty.array[:,:,i,j],skew_bin_facs)[datagood].flatten()
					spice_xya = np.array([xshift[datagood].flatten(),yshift[datagood].flatten()]).T
					dat_lndi = bindown2(lndi(spice_xya, dat)(spice_xa0,spice_ya0),skew_bin_facs)/np.prod(skew_bin_facs)
					err_lndi = bindown2(lndi(spice_xya, err)(spice_xa0,spice_ya0),skew_bin_facs)/np.prod(skew_bin_facs)
					err_lndi[np.isnan(dat_lndi)]=parm.meta['BAD_ERR']; dat_lndi[np.isnan(dat_lndi)]=0
					window[linkey][parmkey].uncertainty.array[:,:,i,j] = err_lndi
					window[linkey][parmkey].data[:,:,i,j] = dat_lndi

	return window

# Perform the full correction on a set of SPICE data with header
# do line fits, deskew them, and return both results:
def full_correction(spice_dat, spice_hdr, xlshift, ylshift, **kwargs):

	skew_bin_facs = kwargs.get('skew_bin_facs',[3,3]); spice_sdev_guess=kwargs.get('spice_sdev_guess',0.1)
	fitter = kwargs.get('fitter',lsq_fitter); lcen = kwargs.get('lcen')
	spice_dx, spice_dy, spice_dl = spice_hdr['CDELT1'],spice_hdr['CDELT2'],10*spice_hdr['CDELT3']
	spice_wl0 = 10*spice_hdr['CRVAL3']-spice_dl*spice_hdr['CRPIX3']
	spice_la = spice_wl0+spice_dl*np.arange(spice_dat.shape[2],dtype=np.float64)
	# Offseting the interpolated points is important. Otherwise there will be ridges
	# in the variance at xlshift=0 and ylshift=0 because that results in some points
	# with no interpolation which in turn have less interpolation smoothing and there
	# for higher variance. 
	offsets = kwargs.get('offsets',[[0.0,0.0]])
	if(lcen is None): lcen = np.mean(spice_la)

	# Apply the skew correction with the specified x shift and yshift
	spicedat_skew = skew_correct(spice_dat, spice_hdr, xlshift, ylshift, lambdas=spice_la,
								 lcen=lcen, skew_bin_facs=skew_bin_facs, offsets=offsets)
	spiceerr_skew = get_spice_err(spicedat_skew, spice_hdr, verbose=False)
	spice_skew_fit_mask, spice_skew_fit_err = get_mask_errs(spicedat_skew, 0.2, error_cube=spiceerr_skew)

	# Find the main range of the the spice data in y and mask out any missing data:
	ymin, ymax, band1_ymin, band1_ymax, band2_ymin, band2_ymax, signal_windowed, signal_cube = get_spice_data_yrange(spicedat_skew)
	spice_skew_fit_mask[:,0:ymin,:] = 1; spice_skew_fit_mask[:,ymax:,:] = 1
	centers, lines = check_for_waves(spice_la, kwargs.get('linelist',None))
	nlines = len(centers)
	ndof = np.sum(np.logical_not(spice_skew_fit_mask),axis=2) - (3*nlines+1)
	# Make a plot of the good range of the data:
	if(kwargs.get('do_yrange_check',False)):
		make_yrange_check_plot(spice_dat, ndof>0, spice_hdr, ymin, ymax, band1_ymin, band1_ymax, band2_ymin,
							band2_ymax, signal_windowed, signal_cube, plot_dir=kwargs.get('yrange_plot_dir'))
	
	# Fit the spectral lines in the data:
	fit_results = list(fitter(spicedat_skew, spiceerr_skew, spice_la, spice_skew_fit_mask, 
						 spice_sdev_guess, linelist=kwargs.get('linelist',None), cenbound_fac=0.0, nthreads=kwargs.get('nthreads'), verbose=False))
	fit_results.append(spice_hdr)
	window_skewed = linefits([fit_results])

	if(kwargs.get('do_deskew',True)):
		window_out = linefits()
		for key in window_skewed: 
			window_out[key] = deskew_linefit_window(window_skewed[key], xlshift, ylshift,
													skew_bin_facs=skew_bin_facs, lcen=np.mean(spice_la))
	else:
		window_out = window_skewed
	return {'dat_skew':spicedat_skew, 'err_skew':spiceerr_skew, 'linefits':window_out, 'xlshift':xlshift,
			'ylshift':ylshift, 'kwargs':kwargs,	'hdr':spice_hdr, 'centers':centers, 'lines':lines, 
			'mask':spice_skew_fit_mask, 'waves':spice_la, 'offsets':offsets, 'ymin':ymin, 'ymax':ymax}
