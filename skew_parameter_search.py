import os, numpy as np; from scipy.optimize import least_squares
from skew_correction import skew_correct, deskew_linefit_window

def simple_lls(dat, err, funcs_in): # Simple LLS fit of funcs with linear coeffs to data
	nf = len(funcs_in)
	nd = dat.size
	rmatp = np.zeros([nf,nd])
	for i in range(0,nf): rmatp[i] = (funcs_in[i]/err).flatten()
	amat = rmatp.dot(rmatp.T)
	bvec = rmatp.dot((dat/err).flatten())
	return np.linalg.inv(amat).dot(bvec)

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

def dopp_var(dopp,vmin,vmax,mask):
	dscal = np.clip(2*(dopp-(vmin+vmax)*0.5)/(vmax-vmin),-1,1)
	return np.nanstd(mask*dscal)

def check_shift(spice_dat, spice_hdr, xlshift, ylshift, line_waves, line_names, **kwargs):
	
	skew_bin_facs = kwargs.get('skew_bin_facs',[3,3]); spice_sdev_guess=kwargs.get('spice_sdev_guess',0.1)
	fitter = kwargs.get('fitter',lsq_fitter); lcen = kwargs.get('lcen')
	spice_dx, spice_dy, spice_dl = spice_hdr['CDELT1'],spice_hdr['CDELT2'],10*spice_hdr['CDELT3']
	spice_wl0 = 10*spice_hdr['CRVAL3']-spice_dl*spice_hdr['CRPIX3']
	spice_la = spice_wl0+spice_dl*np.arange(spice_dat.shape[2],dtype=np.float64)
	if(lcen is None): lcen = np.mean(spice_la)

	spicedat_skew = skew_correct(spice_dat, xlshift, ylshift, spice_dx, spice_dy, spice_dl, spice_wl0, 
								 lambdas=spice_la, lcen=lcen, skew_bin_facs=skew_bin_facs)
	spiceerr_skew = get_spice_err(spicedat_skew, spice_hdr, verbose=False)
	spice_skew_fit_mask, spice_skew_fit_err = get_mask_errs(spicedat_skew, 0.2, error_cube=spiceerr_skew)
	
	fit_results = list(fitter(spicedat_skew, spiceerr_skew, spice_la, spice_skew_fit_mask, 
						 line_waves, spice_sdev_guess, line_names, cenbound_fac=0.0))
	fit_results.append(spice_hdr)
	window_skewed = linefits([fit_results])

	window_deskewed = linefits()
	for key in window_skewed: 
		window_deskewed[key] = deskew_linefit_window(window_skewed[key], xlshift, ylshift, 
												   skew_bin_facs=skew_bin_facs, lcen=np.mean(spice_la))

	var = 0
	for key in window_deskewed[0]:
		dopp = window_deskewed[0][key].get('centers')
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
	print(xshift, yshift, var)
	return xshift, yshift, var

import subprocess
import multiprocessing

class shift_holder(object):
	def __init__(self, spice_dat, spice_hdr, line_waves, line_names, fitter_name, **kwargs):
		self.dat = spice_dat
		self.hdr = spice_hdr
		self.line_waves = line_waves
		self.line_names = line_names
		self.fitter_name = fitter_name
		self.kwargs = kwargs
		self.save_dir = kwargs.get('save_dir',os.getcwd())
		self.save_file = kwargs.get('save_file','shift_vars_'+self.hdr['filename']+self.hdr['EXTNAME']+self.fitter_name+'.npz')
		save_path = os.path.join(self.save_dir,self.save_file)
		self.discretization = kwargs.get('discretization',1000)
		self.valdict = {}
		if(kwargs.get('noload') is None and os.path.exists(save_path)): self.load()

	def get(self,shifts):
		return self.valdict.get(self.make_key(shifts)[0])

	def make_key(self, shifts):
		xind = np.round(shifts[0]*self.discretization).astype(np.int64)
		yind = np.round(shifts[1]*self.discretization).astype(np.int64)
		xshift, yshift = xind/self.discretization, yind/self.discretization
		key = str(xind)+','+str(yind)
		return key, xshift, yshift

	def set(self,args):
		args = np.atleast_2d(args)
		for arg in args:
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

def search_shifts_mp(spice_dat, spice_hdr, xshifts, yshifts, line_waves, line_names, fitter, **kwargs):
	kwargs['fitter'] = fitter
	fitter_name = kwargs['fitter'].__name__
	shift_vars = kwargs.get('shift_vars',shift_holder(spice_dat, spice_hdr, line_waves, line_names, fitter_name, **kwargs))
	xs_flat, ys_flat = xshifts.flatten(), yshifts.flatten()
	packages = []
	for i in range(0,len(xs_flat)):
		if(shift_vars.get([xs_flat[i],ys_flat[i]]) is None):
			key, xs, ys = shift_vars.make_key([xs_flat[i],ys_flat[i]])
			packages.append([spice_dat, spice_hdr, xs, ys, line_waves, line_names, kwargs])
	if __name__ == '__main__':
		#for package in packages: shift_vars.set(shift_runner(package))
		pool = multiprocessing.Pool(9)
		r = pool.map_async(shift_runner, packages, callback=shift_vars.set)
		r.wait()
	return shift_vars
