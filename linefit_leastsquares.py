import numpy as np; from scipy.optimize import least_squares

def lsq_fitter(datacube, errorcube, waves, dat_mask, centers_in, sig_guesses, line_names_in,
			n_status=8, ndof_min=2, noisefloor=None, cenbound_fac = 0.05, widbound_fac=0.25,
			cenbounds=None, verbose=False, xscale_rad=0.1, xscale_wav=0.001, bad_err=-65536,
			fg_param_types = ['WAVELENGTH','RADIANCE','WAVELENGTH'], bg_param_types=['RADIANCE'],
			fg_param_units = ['W/m2/sr','nm','nm'], bg_param_units = ['W/m2/sr']):

	centers, line_names = [[],[]]
	for i in range(0,len(centers_in)):
		if((centers_in[i] > waves[0]) and (centers_in[i] < waves[-1])):
			centers.append(centers_in[i]); line_names.append(line_names_in[i])

	if(np.isscalar(sig_guesses)): sig_guesses = sig_guesses+np.zeros(len(centers))
	if(verbose): print('Fitting ', line_names)
	
	nl, nw, dw = len(centers), len(waves), (waves[-1]-waves[0])
	nx, ny, nf = datacube.shape[0], datacube.shape[1], 3*nl+1
	fits, fit_errs = np.zeros([nx,ny,nf+1]), bad_err*np.ones([nx,ny,nf+1])
	fg_param_names = ['amplitudes','centers','sigmas']

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

	for i in range(0,nx):
		for j in range(0,ny):
			dat, err, msk = datacube[i,j,:], errorcube[i,j,:], np.logical_not(dat_mask[i,j,:])
			# Each subwindow must have at least 3 wavelength points present in order to make a fit:
			mask_check = np.prod([np.sum(msk[subwin[0]:subwin[1]]) >= 3 for subwin in subwins])
			ndof = np.sum(msk) - nf
			if(mask_check and ndof >= ndof_min):
				# Set up bounds and initial guess:
				cont = np.min(dat[msk])
				for k in range(0,nl):
					swdat = dat[subwins[k,0]:subwins[k,1]][msk[subwins[k,0]:subwins[k,1]]]
					swwav = waves[subwins[k,0]:subwins[k,1]][msk[subwins[k,0]:subwins[k,1]]]
					guess[3*k] = np.max(swdat); guess[3*k+1] = swwav[np.argmax(swdat)]
					ubounds[3*k] = guess[3*k] + 5*np.max(err[subwins[k,0]:subwins[k,1]][msk[subwins[k,0]:subwins[k,1]]])
				lbounds[-1]=cont-3*noisefloor; ubounds[-1]=np.max(dat[msk]); guess[-1]=cont
				
				# Run the fitter:
				guess_final = np.clip(guess,lbounds,ubounds)
				dat, err, wvl = dat[msk], err[msk], waves[msk]
				resid = resid_evaluator(wvl,dat,err,multi_gaussian_profile)
				solution = least_squares(resid.get, guess_final, bounds=(lbounds,ubounds), x_scale=xscales)
				
				jac_inv_var = 2*np.sum(solution['jac']**2,axis=0)*ndof/nf # Error estimate based on Jacobian
				if(np.prod(jac_inv_var > 0)): # Assign fits, chi squared, and uncertainties to output:
					fits[i,j,0:nf] = solution['x']; fits[i,j,nf] = solution['cost']
					fit_errs[i,j,0:nf] = (1/jac_inv_var)**0.5; fit_errs[i,j,nf] = np.sqrt(2*ndof)/ndof

		if(verbose and (np.mod(i,np.round(nx/n_status).astype(np.int32))==0 or i==nx-1)): 
			print(i, 'of', nx, 'fraction of lines fit =', np.mean(fit_errs[i,:,-1] != bad_err))

	return fits, fit_errs, data_names, data_types, data_units, param_names

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
