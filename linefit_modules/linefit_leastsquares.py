import subprocess, multiprocessing, numpy as np
from scipy.optimize import least_squares

# Check to see which lines are present in a window
def check_for_waves(waves,linelist=None):
	if(linelist is None): linelist=default_linelist
	line_names_in = list(linelist.keys())
	centers_in = [linelist[name] for name in line_names_in]
	centers, line_names = [[],[]]
	for i in range(0,len(centers_in)):
		if((centers_in[i] > waves[0]) and (centers_in[i] < waves[-1])):
			centers.append(centers_in[i]); line_names.append(line_names_in[i])
			
	return centers, line_names

# Names and wavelengths of the standard lines to search:
default_linelist = {'Ar VIII+S III 700':700.3, 'O III 703':702.9, 'O III 704':703.9, 'Mg IX 706':706.0,
			'O II 718':718.5, 'S IV 745':744.9, 'S IV 748':748.4, 'S IV 750':750.2,
			'O V 759':758.7, 'S IV+O V 759':759.4, 'O V 760':760.3, 'O V 762':762.0,
			'N IV 765':765.1, 'Ne VIII 770':770.4, 'Mg VIII 772':772.3, 'Ne VIII 780':780.3,
			'S V 786':786.5, 'O IV 787':787.7, 'O IV 790':790.1, 'Ly Gamma 972':972.5,
			'C III 977':977.0, 'O I +- Na VI 989':988.7, 'N III 990':989.8, 'N III 992':991.6,
			'H I (+ O I) 1025':1025.7, 'O I 1027':1027.4, 'O VI 1032':1031.9, 'C II 1036':1036.5,
			'O VI 1037':1037.6}

# Main algorithm to run to fit lines. Identifies lines present, sets up the bounds
# for the fitting, runs the actual fitting, and returns results. Can run multi-threaded.
# default is 8 threads. Set single_thread to True for single-threaded run. Required inputs,
# in order:
#  - datacube: cube of SPICE data, with shape nx, ny, nlambda.
#  - errorcube: cube of errors for spice data, same shape as datacube -- see util.get_spice_err.
#               or official routine from sospice package.
#  - waves: wavelengths for the lambdas in the cube, length nlambda. Estimate from header e.g.,
#           spice_wl0 = 10*spice_hdr['CRVAL3']-spice_hdr['CDELT3']*spice_hdr['CRPIX3']
#	        waves = spice_wl0+spice_hdr['CDELT3']*np.arange(datacube.shape[2],dtype=np.float64)           
#  - dat_mask: mask identifying bad data, same shape as datacube -- see util.get_mask_errs.
#  - centers_in: Nominal line center wavelengths to attempt to fit for. Will check 
#                to see if each of these lines is within the wavelength window.
#  - line_names_in: names of lines to correspond with centers_in.
def lsq_fitter(datacube, errorcube, waves, dat_mask, sig_guesses, linelist=None,
			n_status=8, ndof_min=2, noisefloor=None, cenbound_fac = 0.05, widbound_fac=0.25,
			cenbounds=None, verbose=False, xscale_rad=0.1, xscale_wav=0.001, bad_err=-65536,
			fg_param_types = ['RADIANCE','WAVELENGTH','WAVELENGTH'], bg_param_types=['RADIANCE'],
			wav_unit='Angstrom',rad_unit='W/m2/sr', nthreads=None, single_thread=False):

	if(nthreads is None): nthreads=8
	centers, line_names = check_for_waves(waves, linelist=linelist)

	if(np.isscalar(sig_guesses)): sig_guesses = sig_guesses+np.zeros(len(centers))
	if(verbose): print('Fitting ', line_names)
	
	nl, nw, dw = len(centers), len(waves), (waves[-1]-waves[0])
	nx, ny, nf = datacube.shape[0], datacube.shape[1], 3*nl+1
	fits, fit_errs = np.zeros([nx,ny,nf+1]), bad_err*np.ones([nx,ny,nf+1])
	fg_param_names = ['amplitudes','centers','sigmas']
	fg_param_units = [rad_unit+'/'+wav_unit,wav_unit,wav_unit]
	bg_param_units = [rad_unit+'/'+wav_unit]

	param_names, data_names, data_types, data_units = [[],[],[],[]]
	for i in range(0,nl):
		for j in range(0,len(fg_param_names)):
			param_names.append(fg_param_names[j]); data_names.append(line_names[i])
			data_types.append(fg_param_types[j]); data_units.append(fg_param_units[j])

	param_names.append('continuum'); data_names.append('BACKGROUND')
	data_types.append(bg_param_types[0]); data_units.append('bg_param_units'[0])
	param_names.append('chi2'); data_names.append('CHI2')
	data_types.append('RESIDUAL'); data_units.append('')
	
	# Set characteristic scales for amplitude (radiance) and wavelength-like parameters:
	xscales = np.zeros(nf); xscales[-1] = xscale_rad
	for i in range(0,nl): xscales[3*i:3*(i+1)] = [xscale_rad, xscale_wav, xscale_wav]
	if(noisefloor is None): noisefloor = np.min(errorcube[np.logical_not(dat_mask)])

	if(cenbounds is None): # Figure out where the boundaries should be, if they're not supplied:
		csort = np.argsort(centers)
		cpad = np.hstack([np.min(waves),np.sort(np.array(centers).flatten()),np.max(waves)])
		cenbounds, cb_pad = np.zeros([nl,2]), np.zeros([nl+2,2])
		for i in range(1,nl+2): cb_pad[i,0] = 0.5*(cpad[i]+cpad[i-1]) + cenbound_fac*(cpad[i]-cpad[i-1])
		for i in range(0,nl+2-1): cb_pad[i,1] = 0.5*(cpad[i+1]+cpad[i]) - cenbound_fac*(cpad[i+1]-cpad[i])
		for i in range(0,nl): cenbounds[csort[i],:] = cb_pad[i+1,:]
	
	subwins, wave_indices = np.zeros([nl,2],dtype=np.int32), np.arange(nw,dtype=np.int32)
	for i in range(0,nl): # Find subwindow indices for each boundary
		inbounds = wave_indices[(waves >= cenbounds[i][0])*(waves <= cenbounds[i][1])]
		subwins[i,:] = np.min(inbounds),np.max(inbounds)

	lbounds, ubounds, guess = np.zeros(nf),np.zeros(nf),np.zeros(nf)
	for i in range(0,nl): # Fitter boundaries that don't change between pixels:
		lbounds[3*i+1], ubounds[3*i+1] = cenbounds[i]
		lbounds[3*i+2], ubounds[3*i+2] = waves[1]-waves[0], widbound_fac*dw
		guess[3*i+2] = sig_guesses[i]

	packages = []
	for i in range(0,nx):
		packages.append([i, nl, datacube, errorcube, dat_mask, subwins, waves, noisefloor, lbounds, ubounds, guess, xscales, bad_err, ndof_min, widbound_fac])
	results = []
	if(single_thread):
		for pkg in packages: results.append(lsq_fitter_mp_runner(pkg))
	else:
		pool = multiprocessing.Pool(nthreads)
		r = pool.map_async(lsq_fitter_mp_runner, packages, callback=results.append)
		r.get()
	for i in range(0,nx):
		fits[i] = results[0][i][0]
		fit_errs[i] = results[0][i][1]

	return fits, fit_errs, data_names, data_types, data_units, param_names

# This runs the fitter for each raster line, using scipy.optimize.least_squares:
def lsq_fitter_mp_runner(package):
	i, nl, datacube, errorcube, dat_mask, subwins, waves, noisefloor, lbounds, ubounds, guess, xscales, bad_err, ndof_min, widbound_fac = package
	nw, dw = len(waves), (waves[-1]-waves[0])
	nx, ny, nf = datacube.shape[0], datacube.shape[1], 3*nl+1
	fits, fit_errs = np.zeros([ny,nf+1]), bad_err*np.ones([ny,nf+1])
	for j in range(0,ny): # Loop each pixel in the raster line
		dat, err, msk = datacube[i,j,:], errorcube[i,j,:], np.logical_not(dat_mask[i,j,:])
		# Each subwindow must have at least 3 wavelength points present in order to make a fit:
		mask_check = np.prod([np.sum(msk[subwin[0]:subwin[1]]) >= 3 for subwin in subwins])
		ndof = np.sum(msk) - nf
		if(mask_check and ndof >= ndof_min):
			# Set up bounds and initial guess:
			cont = np.min(dat[msk])
			lbounds[-1]=cont-3*noisefloor; ubounds[-1]=np.max(dat[msk]); guess[-1]=cont
			for k in range(0,nl):
				swdat = dat[subwins[k,0]:subwins[k,1]][msk[subwins[k,0]:subwins[k,1]]]
				swwav = waves[subwins[k,0]:subwins[k,1]][msk[subwins[k,0]:subwins[k,1]]]
				guess[3*k] = np.max(swdat); guess[3*k+1] = swwav[np.argmax(swdat)]
				ubounds[3*k] = guess[3*k] + 5*np.max(err[subwins[k,0]:subwins[k,1]][msk[subwins[k,0]:subwins[k,1]]])
				lbounds[3*k+2], ubounds[3*k+2] = waves[1]-waves[0], widbound_fac*dw # Shouldn't be changing?
			
			if(np.prod(ubounds > lbounds) > 0):
				# Run the fitter:
				guess_final = np.clip(guess,lbounds+0.001*(ubounds-lbounds),ubounds-0.001*(ubounds-lbounds))
				dat, err, wvl = dat[msk], err[msk], waves[msk]
				resid = resid_evaluator(wvl,dat,err,multi_gaussian_profile)
				try:
					solution = least_squares(resid.get, guess_final, bounds=(lbounds,ubounds), x_scale=xscales)			
					jac_inv_var = 2*np.sum(solution['jac']**2,axis=0)*ndof/nf # Error estimate based on Jacobian
					if(np.prod(jac_inv_var > 0)): # Assign fits, chi squared, and uncertainties to output:
						fits[j,0:nf] = solution['x']; fits[j,nf] = solution['cost']
						fit_errs[j,0:nf] = (1/jac_inv_var)**0.5; fit_errs[j,nf] = np.sqrt(2*ndof)/ndof
				except ValueError:
					print('ValueError in lsq_fitter_mp_runner at '+str(i)+', '+str(j))
					print('Lower Bound: ',lbounds)
					print('Guess: ',guess)
					print('Upper Bound: ',ubounds)

	return fits, fit_errs

# Multiple Gaussian profile with a constant background:
def multi_gaussian_profile(waves,parms):
	nlines = round((len(parms)-1)/3); profile = parms[-1]+0*waves
	for i in range(0,nlines):
		profile += parms[3*i]*np.exp(-0.5*((waves-parms[3*i+1])/parms[3*i+2])**2)
	return profile

class resid_evaluator(object):
	def __init__(self,waves,dat,err,profile):
		self.waves,self.dat,self.err,self.profile = waves,dat,err,profile		
	def get(self,parms):
		return (self.dat-self.profile(self.waves,parms))/self.err
