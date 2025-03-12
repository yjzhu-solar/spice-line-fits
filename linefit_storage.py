# These are routines for storing the results of spectral line fitting
# in fits files and read them back into an organized internal Python
# representation.

import os, copy, numpy as np; from astropy.io import fits; 
from astropy.nddata import NDData, StdDevUncertainty

# These are the special names that the storage algorithm looks for in fits
# files in order to turn them from the flat fits structure into
# the more nested internal representation:
uncertainty_name_postfix='ERRORS'; uncertainty_dat_type='SPECTRAL_FIT_PARAMETER_ERRORS'
parameter_name_key='PRM_NAME'; line_name_key='LIN_NAME'; win_name_key='WIN_NAME'

class iterdict(dict):
	def __getitem__(self,key):
		if(str(key)==key): return super().__getitem__(key)
		else: return super().__getitem__(list(self.keys())[key])

	def __setitem__(self,key,value):
		if(str(key)==key): return super().__setitem__(key,value)
		else: return super().__setitem__(list(self.keys())[key],value)

# Pass either filename='*.fits' or lists of arguments to create_spice_linefit_window on init:
class linefits(iterdict):
	def __init__(self, *args, **kwargs): 
		iterdict.__init__(self)
		if(len(args) > 0):
			if(isinstance(args[0],list)): 
				for arg in args[0]: self.from_raw(arg,**kwargs)
			else: self.update(args[0])
		if(kwargs.get('filename',0)): 
			self.load((kwargs.get('filename')))

	# Could probably stand to have a pretty print method here...

	def save(self,filename):
		write_spice_linefit_file(filename,self)
		
	def load(self,filename):
		self.update(read_spice_linefit_file(filename))
		
	def to_raw(self,*args,**kwargs):
		return [linefit_window_raws(self[win]) for win in self]
		
	def from_raw(self,args,**kwargs):
		if(isinstance(args,list)): self.update(create_linefit_window(*args,**kwargs))
		else: self.update(args)

# This creates a dict-like object reflecting the line fits in a spectral window. In the example
# included below the data and uncertainty arrays contain line center, amplitude, width,
# continuum level, and chi squared for a fit of the C III 977 peak SPICE window (omits Lyman
# Gamma). The arguments in this case would be as follows:
#	 data: ndarray with dimensions nx, ny, nc; nc is number of fit parameters in the window.
#	 uncertainties: ndarray with same dimension as data containing uncertainties in data
#	 data_names: line/data component identifier, of length nc. e.g., 
#			['C III 977','C III 977','C III 977','BACKGROUND','CHI2']
#			Note duplication across, for example, parameters of the same line.
#	 data_types: Data type as in fits BTYPE, of length nc. e.g.,
#			['WAVELENGTH','RADIANCE','WAVELENGTH','RADIANCE','RESIDUAL']
#	 data_units: Data units as in fits BUNIT, of length nc. e.g., 
#			['nm', 'W/m2/sr', 'nm', 'W/m2/sr', '']
#	 param_names: Names of each parameter returned by line fitting, of length nc, e.g.,
#			['centers', 'amplitudes', 'sigmas', 'continuum', 'chi2']
#	 window_header: The header of the spice fits data window in question
def create_linefit_window(data, uncertainties, data_names, data_types, data_units,
								param_names, window_header, bad_err=None):

	nx, ny, nc = data.shape[0], data.shape[1], data.shape[2]
	linefits = iterdict()
	if(bad_err is None): bad_err = window_header.get('BAD_ERR',-65536)
	for i in range(0,nc):
		if(linefits.get(data_names[i]) is None): linefits[data_names[i]] = iterdict()
		dat = np.expand_dims(data[:,:,i],(2,3)) # Retain wavelength & time axes bc header
		err = StdDevUncertainty(np.expand_dims(uncertainties[:,:,i],(2,3)))
		# Wavelength axis is not removed but changed to reflect the line fits span the window:
		# This may need to be fixed to reflect how fits standard defines CRPIX (unit offset,
		# first pixel runs from 0.5 to 1.5 per Greisen & Calabretta A&A 395 1061
		hdr = copy.deepcopy(window_header); hdr['BAD_ERR'] = bad_err
		hdr['CRVAL3'] = hdr['CRVAL3'] + (0.5*hdr['NAXIS3']-(hdr['CRPIX3']-0.5))*hdr['CDELT3']
		hdr['NBIN3']=hdr['NBIN3']*window_header['NAXIS3']; hdr['CRPIX3']=1; hdr['NAXIS3']=1
		hdr['BTYPE'] = data_types[i]; hdr['BUNIT'] = data_units[i]; 
		if(hdr.get('UCD') is not None): hdr.pop('UCD')
		hdr[line_name_key] = data_names[i]; hdr[parameter_name_key] = param_names[i]
		linefits[data_names[i]][param_names[i]] = NDData(dat, uncertainty=err, meta=hdr)
	
	return iterdict({window_header['EXTNAME']:linefits})

# This turns the window into a 'raw' Python format with the constituent arrays
# split up. It is the opposite of create_linefit_window.
def linefit_window_raws(window):
	
	data, errs, data_names, data_types, data_units, param_names = [[],[],[],[],[],[]]
	for linkey in window:
		line = window[linkey]
		for prmkey in line:
			data.append(line[prmkey].data.squeeze())
			errs.append(line[prmkey].uncertainty.array.squeeze())
			data_names.append(linkey); param_names.append(prmkey)
			data_types.append(line[prmkey].meta['BTYPE'])
			data_units.append(line[prmkey].meta['BUNIT'])
	output_header = copy.deepcopy(window[0][0].meta)
	output_header.pop('BTYPE'); output_header.pop('BUNIT');
	
	return [np.array(data).transpose([1,2,0]),np.array(errs).transpose([1,2,0]), data_names,
			data_types, data_units, param_names, output_header]

# Update a fits file with a data array and header. Suprisingly
# astropy fits doesn't support this very well...
def update_fits(filename, data, header):
	# Check to see if the file already exists:
	if(os.path.exists(filename)):
		hdul = fits.open(filename,mode='update')
		names = [h.header['EXTNAME'] for h in hdul]
		if(header['EXTNAME'] in names): index = names.index(header['EXTNAME']) 
		else: index = len(hdul) # Just put the new index at the end
		hdul.insert(index, fits.ImageHDU(data=data.T, header=header))
		if(header['EXTNAME'] in names): hdul.pop(index+1) # We replaced index, pop old one
		hdul.flush(); hdul.close()
	else: # Otherwise, we can just use append:
		fits.append(filename, data.T, header=header)

# Write spice spectral line fit information to a fits file:
def write_spice_linefit_file(filename, window_fits):
	for win in window_fits:
		for lin in window_fits[win]:
			for prm in window_fits[win][lin]:
				hdr = copy.deepcopy(window_fits[win][lin][prm].meta)
				hdr['WIN_NAME'] = hdr['EXTNAME']
				hdr['EXTNAME'] = win+' '+lin+' '+prm
				dat = window_fits[win][lin][prm].data
				update_fits(filename, dat, hdr)

				hdr['PRM_NAME'] = hdr['PRM_NAME']+uncertainty_name_postfix
				hdr['EXTNAME'] = win+' '+lin+' '+prm+' '+uncertainty_name_postfix
				hdr['DAT_TYPE'] = uncertainty_dat_type
				err = window_fits[win][lin][prm].uncertainty.array
				update_fits(filename, err, hdr)

# Read spice spectral line fit information from a fits file:
def read_spice_linefit_file(filename):
	windows = iterdict()
	hdul = fits.open(filename)
	for i in range(0,len(hdul)):
		dat,hdr = hdul[i].data,hdul[i].header
		win = hdr['WIN_NAME']; lin = hdr['LIN_NAME']; prm = hdr['PRM_NAME']
		if(hdr.get('DAT_TYPE') != uncertainty_dat_type):
			err=StdDevUncertainty(hdul[win+' '+lin+' '+prm+' '+uncertainty_name_postfix].data.T)
			if(windows.get(win) is None): windows[win]=iterdict()
			if(windows[win].get(lin) is None): windows[win][lin]=iterdict()
			windows[win][lin][prm] = NDData(dat.T, uncertainty=err, meta=hdr)
	hdul.close()
				
	return windows
