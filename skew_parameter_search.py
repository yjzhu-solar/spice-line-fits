# This file contains codes to search for optimal x and y shifts as a function of lambda,
# To remove SPICE Doppler artifacts. Intermediate results can be stored to a csv file
# and reloaded with the shift_holder objects. The search_shifts routine will check
# a set of shifts, while refine_points will take a preset group of searched points
# and produce a new set of search points in the vicinity of the best ones.
# The goodness-of-fit is taken to be the residual Doppler shift across the image
# after a linear trend has been removed. See example notebook for usage.

import os, copy, numpy as np, matplotlib.pyplot as plt
from scipy.optimize import least_squares
from skew_correction import skew_correct, deskew_linefit_window, full_correction
from linefit_leastsquares import lsq_fitter, check_for_waves
from util import get_mask_errs, get_spice_err, get_spice_data_yrange, make_yrange_check_plot
from linefit_storage import linefits
from astropy.io import fits

def simple_lls(dat, err, funcs_in): # Simple LLS fit of funcs with linear coeffs to data
	nf = len(funcs_in)
	nd = dat.size
	rmatp = np.zeros([nf,nd])
	for i in range(0,nf): rmatp[i] = (funcs_in[i]/err).flatten()
	amat = rmatp.dot(rmatp.T)
	bvec = rmatp.dot((dat/err).flatten())
	return np.linalg.inv(amat).dot(bvec)

# The SPICE Doppler tend to contain a trend across the image field.
# Since this will impact the Doppler variance we use to determine which
# shift is optimal we fit linear trends in x and y and remove them:
def detrend_dopp(dopp_ndd): # Remove linear spatial trend in x and y from Doppler
	dopp = dopp_ndd.data.squeeze()
	dopp_err = dopp_ndd.uncertainty.array.squeeze()

	snr_th = np.abs(dopp) > 2*np.abs(dopp_err)
	mask = snr_th*(dopp_err > 0)
	nx,ny = dopp.shape
	x0 = np.ones([nx,ny])
	x1, x2 = np.indices([nx,ny])
	x1 = x1-0.5*nx; x2 = x2-0.5*ny
	mask = (np.isnan(dopp)==False)*snr_th

	cvec = simple_lls(dopp[mask], dopp_err[mask], [x0[mask],x1[mask],x2[mask]])

	dopp_detrend = copy.deepcopy(dopp) - x1*cvec[1] - x2*cvec[2] 
	return dopp_detrend

# Compute the variance/standard deviation of the Doppler,
# scaled by vmin and vmax:
def dopp_var(dopp,vmin,vmax,mask):
	dscal = np.clip(2*(dopp-(vmin+vmax)*0.5)/(vmax-vmin),-1,1)
	return np.nanstd(mask*dscal)

# Try a specified x and y shift to see what its variance is:
def check_shift(spice_dat, spice_hdr, xlshift, ylshift, **kwargs):
	
	# skew_bin_facs = kwargs.get('skew_bin_facs',[3,3]); spice_sdev_guess=kwargs.get('spice_sdev_guess',0.1)
	# fitter = kwargs.get('fitter',lsq_fitter); lcen = kwargs.get('lcen')
	# spice_dx, spice_dy, spice_dl = spice_hdr['CDELT1'],spice_hdr['CDELT2'],10*spice_hdr['CDELT3']
	# spice_wl0 = 10*spice_hdr['CRVAL3']-spice_dl*spice_hdr['CRPIX3']
	# spice_la = spice_wl0+spice_dl*np.arange(spice_dat.shape[2],dtype=np.float64)
	# # Offseting the interpolated points is important. Otherwise there will be ridges
	# # in the variance at xlshift=0 and ylshift=0 because that results in some points
	# # with no interpolation which in turn have less interpolation smoothing and there
	# # for higher variance. 
	# offsets = kwargs.get('offsets',[[0.5,0.5]])
	# if(lcen is None): lcen = np.mean(spice_la)

	# # Apply the skew correction with the specified x shift and yshift
	# spicedat_skew = skew_correct(spice_dat, spice_hdr, xlshift, ylshift, lambdas=spice_la,
	#							 lcen=lcen, skew_bin_facs=skew_bin_facs, offsets=offsets)
	# spiceerr_skew = get_spice_err(spicedat_skew, spice_hdr, verbose=False)
	# spice_skew_fit_mask, spice_skew_fit_err = get_mask_errs(spicedat_skew, 0.2, error_cube=spiceerr_skew)

	# # Find the main range of the the spice data in y and mask out any missing data:
	# ymin, ymax, band1_ymin, band1_ymax, band2_ymin, band2_ymax, signal_windowed, signal_cube = get_spice_data_yrange(spicedat_skew)
	# spice_skew_fit_mask[:,0:ymin,:] = 1; spice_skew_fit_mask[:,ymax:,:] = 1
	# centers, lines = check_for_waves(line_waves, line_names, spice_la)
	# nlines = len(centers)
	# ndof = np.sum(np.logical_not(spice_skew_fit_mask),axis=2) - (3*nlines+1)
	# # Make a plot of the good range of the data:
	# make_yrange_check_plot(spice_dat, ndof>0, spice_hdr, ymin, ymax, band1_ymin, band1_ymax, band2_ymin,
	#						band2_ymax, signal_windowed, signal_cube, plot_dir=kwargs.get('yrange_plot_dir'))
	
	# # Fit the spectral lines in the data:
	# fit_results = list(fitter(spicedat_skew, spiceerr_skew, spice_la, spice_skew_fit_mask, 
	#					 line_waves, spice_sdev_guess, line_names, cenbound_fac=0.0, nthreads=kwargs.get('nthreads'), verbose=False))
	# fit_results.append(spice_hdr)
	# window_skewed = linefits([fit_results])

	# if(kwargs.get('do_deskew',False)):
	#	window_out = linefits()
	#	for key in window_out: 
	#		window_out[key] = deskew_linefit_window(window_skewed[key], xlshift, ylshift, dokeys=['centers'],
	#												   skew_bin_facs=skew_bin_facs, lcen=np.mean(spice_la))
	# else:
	#	window_out = window_skewed
	correction_results = full_correction(spice_dat, spice_hdr, xlshift, ylshift, **kwargs)

	window_out = correction_results['linefits']	
	# Compute the overall Doppler variance:
	var = 0
	for key in window_out[0]:
		dopp = window_out[0][key].get('centers')
		if(dopp is not None):
			dopp_dt = detrend_dopp(dopp)
			dopp_mask = (dopp.data > 2*dopp.uncertainty.array)*(dopp.uncertainty.array > 0)
			wlcen = np.nanmedian(dopp.data[dopp_mask])
			dopmin,dopmax = wlcen+np.array([-0.1,0.1])
			var += dopp_var(dopp_dt,dopmin,dopmax,dopp_mask.squeeze())

	if(kwargs.get('verbose',False)): print('Checked ', xlshift, ylshift, 'variance=', var)
	return var, dopp_dt

def shift_runner(package):
	xshift, yshift, var = package[2], package[3], check_shift(*package[0:-1],**package[-1])[0]
	return xshift, yshift, var

import subprocess
import multiprocessing

# This holds a set of line shift variables, and allows loading and saving
# and looking up the results of a search. The save format is a simple 
# human readable csv file:
class shift_holder(object):
	def __init__(self, spice_dat, spice_hdr, fitter_name, **kwargs):
		self.dat = spice_dat
		self.hdr = spice_hdr
		self.fitter_name = fitter_name
		self.kwargs = kwargs
		self.save_dir = kwargs.get('save_dir',os.getcwd())
		self.save_file = kwargs.get('save_file','shift_vars_'+self.hdr['filename']+self.hdr['EXTNAME']+self.fitter_name+'.csv')
		self.save_file = self.save_file.replace('/','_')
		save_path = os.path.join(self.save_dir,self.save_file)
		self.discretization = kwargs.get('discretization',1000)
		self.valdict = {}
		if(kwargs.get('noload') is None and os.path.exists(save_path)): self.load()

	def get(self,shifts):
		vals = self.valdict.get(self.make_key(shifts)[0])
		if(vals is not None): vals=np.array(vals)
		return vals
		
	def all_shifts(self):
		return np.array(list(self.valdict.values())).T
	
	def best_shifts(self):
		xs,ys,vals = self.all_shifts()
		return xs[np.argmin(vals)],ys[np.argmin(vals)]

	def make_key(self, shifts):
		xind = np.round(shifts[0]*self.discretization).astype(np.int64)
		yind = np.round(shifts[1]*self.discretization).astype(np.int64)
		xshift, yshift = xind/self.discretization, yind/self.discretization
		key = str(xind)+','+str(yind)
		return key, xshift, yshift

	def set(self,arg_in):
		if('valdict' in dir(arg_in)): args = arg_in.valdict.values()
		else: args = np.atleast_2d(arg_in)
		for arg in args:
			if(len(arg)==3):
				shifts, var = arg[0:2], arg[2]
				key, xs, ys = self.make_key(shifts)
				self.valdict[key] = xs, ys, var

	def save(self,filename=None):
		if(filename is None): filename = self.save_file
		keys = list(self.valdict.keys())
		outarray = np.zeros([3,len(keys)])
		for i in range(0,len(keys)): outarray[:,i] = self.valdict[keys[i]]
		header = ('PSF correction shift variances for ' + self.hdr['filename'] + ' ' + self.hdr['EXTNAME'] + '\n' + 
				   'discretization: '+ str(self.discretization))
		np.savetxt(os.path.join(self.save_dir,filename), outarray.T, header=header, delimiter=',')

	def load(self, filename=None):
		if(filename is None): filename = self.save_file
		self.set(np.loadtxt(os.path.join(self.save_dir,filename), delimiter=','))

# Search a set of x and y shifts for a specified SPICE window with header, and set of lines:
def search_shifts(spice_dat, spice_hdr, xshifts, yshifts, fitter, **kwargs):
	kwargs['fitter'] = fitter
	fitter_name = kwargs['fitter'].__name__
	shift_vars = kwargs.get('shift_vars',shift_holder(spice_dat, spice_hdr, fitter_name, **kwargs))
	xs_flat, ys_flat = xshifts.flatten(), yshifts.flatten()
	packages = []
	for i in range(0,len(xs_flat)):
		if(shift_vars.get([xs_flat[i],ys_flat[i]]) is None):
			key, xs, ys = shift_vars.make_key([xs_flat[i],ys_flat[i]])
			packages.append([spice_dat, spice_hdr, xs, ys, kwargs])
	if(kwargs.get('search_multi_thread',False)):
		pool = multiprocessing.Pool(kwargs.get('search_nthread',8))
		results = []
		r = pool.map_async(shift_runner, packages, callback=results.append)
		r.wait()
		for result in results: shift_vars.set(result)
	else:
		for i in range(0,len(packages)):
			svar = shift_runner(packages[i]) 
			shift_vars.set(svar)
			print(shift_vars.get(svar[0:2]))
	return shift_vars

# Given an initial grid of evaluated shift points, produce a new set of points to check
# which are near the best of the initial evaluated points and on a new grid defined by
# xgrange, ygrange, ngx, and ngy:
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator as lndi
def refine_points(shift_vars, xgrange, ygrange, ngx, ngy, npts_refine, min_input=0.275, min_output=0.1):
	xa = np.array(list(shift_vars.valdict.values()))[:,0]
	ya = np.array(list(shift_vars.valdict.values()))[:,1]
	dat = np.array(list(shift_vars.valdict.values()))[:,2]
	
	xya = np.vstack([xa,ya]).T
	xa0,ya0 = np.array(np.meshgrid(np.linspace(xgrange[0], xgrange[1],ngx),np.linspace(ygrange[0],ygrange[1],ngy))).transpose([0,2,1])
	dat_interp = lndi(xya, dat)(xa0,ya0)

	sort_interp = np.argsort(dat_interp.flatten())
	xsort_interp = xa0.flatten()[sort_interp]
	ysort_interp = ya0.flatten()[sort_interp]

	return xsort_interp[0:npts_refine], ysort_interp[0:npts_refine]

# Searches for the best correction parameters in a specific window of a SPICE file.
# Results are saved to a figure and a csv file. Does an initial search on a coarse grid
# then two grid refinement steps where it searches the neighborhood of the best points
# found at previous steps. Required arguments:
# file_name: the name of the SPICE fits file to check
# win_name: the name of the spectral window within the file to check
# Keywords:
#   xl, xh, yl, yh: range of x and y shifts to check (def: -5 to 5)
#   n0, n1, n2: Grid dimensions at each refinement level (def: 5, 11, 31)
#   n_refine: Number of best-points to check out of the refinement subgrids
#   nthreads: Number of threads to use (default: 8)
# Output is placed in the directory specified by the keyword save_dir (default: spice_shift_vars)
# Save directory needs to have the following subdirectories:
# save: Contains the CSV files
# yrange_plots: diagnostic plots to see if the y range of the observation window is correctly selected
# figs: Contains the final plot showing goodness of fit as function of x and y shift
def search_spice_window(filename, win_name, xl=-5, yl=-5, xh=5, yh=5, n0=5, n1=11, n2=31, 
			n_refine=20, save_dir='../spice_shift_vars/', linelist=None, 
			fitter=lsq_fitter, nthreads=8):

	yrange_plot_dir = os.path.join(save_dir,'yrange_plots')
	shift_save_dir = os.path.join(save_dir,'save')
	shift_plot_dir = os.path.join(save_dir,'figs')

	hdul = fits.open(filename)
	spice_dat, spice_hdr = hdul[win_name].data[0], hdul[win_name].header
	spice_dat = spice_dat.transpose([2,1,0]).astype(np.float32)
	hdul.close()

	spice_dx, spice_dy, spice_dl = spice_hdr['CDELT1'],spice_hdr['CDELT2'],10*spice_hdr['CDELT3']
	spice_wl0 = 10*spice_hdr['CRVAL3']-spice_dl*spice_hdr['CRPIX3']
	spice_la = spice_wl0+spice_dl*np.arange(spice_dat.shape[2],dtype=np.float64)

	centers, lines = check_for_waves(spice_la, linelist=linelist)
	print('Found the following lines in '+win_name+': '+str(lines))

	xs_initial, ys_initial = np.array(np.meshgrid(np.linspace(xl,xh,5),np.linspace(yl,yh,5))).transpose([0,2,1])
	shift_vars = shift_holder(spice_dat, spice_hdr, fitter.__name__, save_dir=shift_save_dir)
	sv_initial = search_shifts(spice_dat, spice_hdr, xs_initial, ys_initial, fitter,
					linelist=linelist, single_thread=True, nthreads=nthreads, offsets=[[0.5,0.5]], do_deskew=False,
					yrange_plot_dir=yrange_plot_dir, shift_vars=shift_vars, do_yrange_check=True)
	shift_vars.set(sv_initial)
	shift_vars.save()

	x_refine, y_refine = refine_points(shift_vars,[xl,xh],[yl,yh], n1, n1, n_refine)
	shift_vars = search_shifts(spice_dat, spice_hdr, x_refine, y_refine, fitter,
					linelist=linelist, single_thread=True, nthreads=nthreads, offsets=[[0.5,0.5]], do_deskew=False,
					yrange_plot_dir=yrange_plot_dir, shift_vars=shift_vars, do_yrange_check=True)
	shift_vars.save()

	x_refine, y_refine = refine_points(shift_vars,[xl,xh],[yl,yh], n2, n2, n_refine)
	shift_vars = search_shifts(spice_dat, spice_hdr, x_refine, y_refine, fitter,
					linelist=linelist, single_thread=True, nthreads=nthreads, offsets=[[0.5,0.5]], do_deskew=False,
					yrange_plot_dir=yrange_plot_dir, shift_vars=shift_vars, do_yrange_check=True)
	shift_vars.save()

	# Reinterpolate the search results to a finer linear grid for ease of plotting:
	from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator as lndi

	xa = np.array(list(shift_vars.valdict.values()))[:,0]
	ya = np.array(list(shift_vars.valdict.values()))[:,1]
	dat = np.array(list(shift_vars.valdict.values()))[:,2]

	include = (np.abs(xa) > 1.0e-5)*(np.abs(ya) > 1.0e-5)

	nx_plot, ny_plot = 41, 41
	xya = np.vstack([xa[include],ya[include]]).T
	xa0,ya0 = np.array(np.meshgrid(np.linspace(xl,xh,nx_plot),np.linspace(yl,yh,ny_plot))).transpose([0,2,1])
	dat_interp = lndi(xya, dat[include])(xa0,ya0)

	dat_interp = lndi(xya, dat[include])(xa0,ya0)

	sort_interp = np.argsort(dat_interp.flatten())
	xsort_interp = xa0.flatten()[sort_interp]
	ysort_interp = ya0.flatten()[sort_interp]

	xl2, xh2 = xl-0.5*(xh-xl)/(nx_plot-1), xh+0.5*(xh-xl)/(nx_plot-1)
	yl2, yh2 = yl-0.5*(yh-yl)/(ny_plot-1), yh+0.5*(yh-yl)/(ny_plot-1)

	labelstr = spice_hdr['DATE-OBS']+' '+spice_hdr['EXTNAME']
	labelstr = labelstr.replace('-','').replace(':','').replace('  ','_').replace(' ','_')
	labelstr = labelstr.replace('/','_')
	print(labelstr)

	fig,axes = plt.subplots(nrows=1,ncols=2,figsize=[16,9])
	plt.suptitle(spice_hdr['DATE-OBS']+' '+spice_hdr['EXTNAME']+': xyshift='+str(np.array([xa[np.argmin(dat)], ya[np.argmin(dat)]])))
	axes[0].imshow(np.clip(np.nansum(spice_dat,axis=2).T,0,None)[150:850,:]**0.5,vmin=0,vmax=(100*np.nanmean(spice_dat))**0.5,aspect=spice_hdr['CDELT2']/spice_hdr['CDELT1'])
	axes[0].set(title='Spectral sum',xlabel='Raster axis @ ypix equivalent -- '+str(spice_hdr['cdelt2'])+'"')
	asdfa = axes[1].imshow(dat_interp.T, extent=[xl2, xh2, yl2, yh2],cmap=plt.get_cmap('gray'))
	axes[1].plot(xa,ya,'P',markersize=10,linewidth=5)
	axes[1].set(title='RMS Doppler variance', xlabel='x shift (arcsecond/angstrom)', ylabel='y shift (arcsecond/angstrom)')
	axes[1].legend(['Sampled points'])
	fig.colorbar(asdfa, ax=axes[1],location='bottom')
	plt.savefig(os.path.join(shift_plot_dir,'varplot_'+labelstr+'.png'))
	plt.close()
	return shift_vars
