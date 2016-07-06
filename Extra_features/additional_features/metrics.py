

import random
import numpy as np
import cv2
import cv2.cv as cv
import pylab as pl
from scipy.signal import convolve2d, wiener, detrend
from skimage import exposure

from scipy.stats import skew

def buildHeader(tmpl,maxT=60):
	""" Build a header

		Build the header for the metrics

		Args:
			tmpl: templateManager object

		Returns:
			header string as csv
	"""
	hdr_ = []
	prefix_ = ['max','xLoc','yLoc']
	for p_ in prefix_:
		for i in range(tmpl.size):
			hdr_.append(p_+'_%07d'%tmpl.info[i]['file'])
	for p_ in prefix_:
		for i in range(tmpl.size):
			hdr_.append(p_+'H_%07d'%tmpl.info[i]['file'])

	# Add time metrics
	for i in range(maxT):
		hdr_ += ['centTime_%04d'%i]
	for i in range(maxT):
		hdr_ += ['bwTime_%04d'%i]
	for i in range(maxT):
		hdr_ += ['skewTime_%04d'%i]
	for i in range(maxT):
		hdr_ += ['tvTime_%04d'%i]

	# Add time metrics
	for i in range(50):
		hdr_ += ['centOops_%04d'%i]

	# Add high frequency metrics
	hdr_ += ['CentStd','AvgBwd','hfCent','hfBwd']

	# Add template 2 metrics
	for p_ in prefix_:
		for i in range(4):
			hdr_.append(p_+'_tmpl2V_%d'%i)

	for p_ in prefix_:
		for i in range(4):
			hdr_.append(p_+'_tmpl2H_%d'%i)

	# Add diffmetrics
	for i in range(4):
			hdr_.append('diffm2_V_%d'%i)
	for i in range(4):
			hdr_.append('diffm2_H_%d'%i)

	hdr_ += ['hfMax','hfMax2','hfMax3']

	return ','.join(hdr_)




def computeMetrics(P, tmpl, bins, maxT, tmpl2_dir):
	""" Compute a bunch of metrics

		Perform template matching and time stats

		Args:
			P: 2-d numpy array
			tmpl: templateManager object
			bins: time bins
			maxT: maximum frequency slice for time stats

		Returns:
			List of metrics
	"""
	tmpl2, xyloc = load_tmpl2(tmpl2_dir)

	Q = slidingWindowV(P,inner=3,maxM=40)
	W = slidingWindowH(P,inner=3,outer=32,maxM=60)
	out = templateMetrics(Q, tmpl)	
	out += templateMetrics(W, tmpl)	
	out += timeMetrics(P,bins,maxM=maxT)
	out += oopsMetrics(P,bins)
	out += highFreqMetrics(P,bins)
	out += template2Metrics(Q, tmpl2, xyloc) # +12 features
	out += template2Metrics(W, tmpl2, xyloc) # +12 features
	out += diffMetrics(Q, tmpl2, maxM=40)	# +4 features
	out += diffMetrics(W, tmpl2, maxM=60)	# +4 features
	return out

def matchTemplate(P, template):
	""" Max correlation and location

		Calls opencv's matchTemplate and returns the
		max correlation and location

		Args:
			P: 2-d numpy array to search
			template: 2-d numpy array to match

		Returns:
			maxVal: max correlation
			maxLoc: location of the max
	"""
	m, n = template.shape
	mf = cv2.matchTemplate(P.astype('Float32'), template, cv2.TM_CCOEFF_NORMED)
	minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(mf)
	return maxVal, maxLoc[0], maxLoc[1]

def slidingWindow(P,inX=3,outX=32,inY=3,outY=64,maxM=50,norm=True):
	""" Enhance the constrast

		Cut off extreme values and demean the image
		Utilize scipy convolve2d to get the mean at a given pixel
		Remove local mean with inner exclusion region

		Args:
			P: 2-d numpy array image
			inX: inner exclusion region in the x-dimension
			outX: length of the window in the x-dimension
			inY: inner exclusion region in the y-dimension
			outY: length of the window in the y-dimension
			maxM: size of the output image in the y-dimension
			norm: boolean to cut off extreme values

		Returns:
			Q: 2-d numpy contrast enhanced
	"""
	Q = P.copy()
	m, n = Q.shape
	
	if norm:
		mval, sval = np.mean(Q[:maxM,:]), np.std(Q[:maxM,:])
		fact_ = 1.5
		Q[Q > mval + fact_*sval] = mval + fact_*sval
		Q[Q < mval - fact_*sval] = mval - fact_*sval
	wInner = np.ones((inY,inX))
	wOuter = np.ones((outY,outX))
	Q = Q - (convolve2d(Q,wOuter,'same') - convolve2d(Q,wInner,'same'))/(wOuter.size - wInner.size)

	return Q[:maxM,:]

def slidingWindowV(P,inner=3,outer=64,maxM = 50,norm=True):
	""" Enhance the constrast vertically (along frequency dimension)

		Cut off extreme values and demean the image
		Utilize numpy convolve to get the mean at a given pixel
		Remove local mean with inner exclusion region

		Args:
			P: 2-d numpy array image
			inner: inner exclusion region 
			outer: length of the window
			maxM: size of the output image in the y-dimension
			norm: boolean to cut off extreme values

		Returns:
			Q: 2-d numpy contrast enhanced vertically
	"""
	Q = P.copy()
	m, n = Q.shape
	if norm:
		mval, sval = np.mean(Q[:maxM,:]), np.std(Q[:maxM,:])
		fact_ = 1.5
		Q[Q > mval + fact_*sval] = mval + fact_*sval
		Q[Q < mval - fact_*sval] = mval - fact_*sval
	wInner = np.ones(inner)
	wOuter = np.ones(outer)
	for i in range(n):
		Q[:,i] = Q[:,i] - (np.convolve(Q[:,i],wOuter,'same') - np.convolve(Q[:,i],wInner,'same'))/(outer - inner)
	# Q = exposure.equalize_hist(Q.astype('Float32'), nbins = 65)
	# Q = detrend(Q.astype('Float32'), axis = 1)
	# Q = wiener(Q.astype('Float32'), 4)
	return Q[:maxM,:]

def slidingWindowH(P,inner=3,outer=32,maxM=50,norm=True):
	""" Enhance the constrast horizontally (along temporal dimension)

		Cut off extreme values and demean the image
		Utilize numpy convolve to get the mean at a given pixel
		Remove local mean with inner exclusion region

		Args:
			P: 2-d numpy array image
			inner: inner exclusion region 
			outer: length of the window
			maxM: size of the output image in the y-dimension
			norm: boolean to cut off extreme values

		Returns:
			Q: 2-d numpy contrast enhanced vertically
	"""
	Q = P.copy()
	m, n = Q.shape
	if norm:
		mval, sval = np.mean(Q[:maxM,:]), np.std(Q[:maxM,:])
		fact_ = 1.5
		Q[Q > mval + fact_*sval] = mval + fact_*sval
		Q[Q < mval - fact_*sval] = mval - fact_*sval
	wInner = np.ones(inner)
	wOuter = np.ones(outer)
	for i in range(maxM):
		Q[i,:] = Q[i,:] - (np.convolve(Q[i,:],wOuter,'same') - np.convolve(Q[i,:],wInner,'same'))/(outer - inner)
	return Q[:maxM,:]

def timeMetrics(P, b,maxM=50):
	""" Calculate statistics for a range of frequency slices

		Calculate centroid, width, skew, and total variation
			let x = P[i,:], and t = time bins
			centroid = sum(x*t)/sum(x)
			width = sqrt(sum(x*(t-centroid)^2)/sum(x))
			skew = scipy.stats.skew
			total variation = sum(abs(x_i+1 - x_i))

		Args:
			P: 2-d numpy array image
			b: time bins 

		Returns:
			A list containing the statistics

	"""
	m, n = P.shape
	cf_ = [np.sum(P[i,:]*b)/np.sum(P[i,:]) for i in range(maxM)]
	bw_ = [np.sum(P[i,:]*(b - cf_[i])*(b - cf_[i]))/np.sum(P[i,:]) for i in range(maxM)]
	sk_ = [skew(P[i,:]) for i in range(maxM)]
	tv_ = [np.sum(np.abs(P[i,1:] - P[i,:-1])) for i in range(maxM)]
	return cf_ + bw_ + sk_ + tv_

def oopsMetrics(P, b,maxM=50):
	""" Oops metrics

		This was supposed to be the centroid of the frequency slices
		as defined in timeMetrics. Fortunate typo that has power
		in discrimination.

		Args:
			P: 2-d numpy array image
			b: time bins
			maxM: max frequency slice to consider

		Returns:
			A list containing the statistics
	"""
	cf_ = [np.sum(P[i,:]*b)/np.sum(P[:,i]) for i in range(maxM)]
	return cf_ 
		
def highFreqTemplate(P, tmpl):
	""" High frequency template matching

		Apply horizontal contrast enhancement and
		look for strong vertical features in the image.
		Cut out the lower frequencies

		Args:
			P: 2-d numpy array image
			tmpl: 2-d numpy array template image

		Returns:
			Maximum correlation as a list
	"""
	Q = slidingWindowH(P,inner=7,maxM=50,norm=True)[38:,:]	
	mf = cv2.matchTemplate(Q.astype('Float32'), tmpl, cv2.TM_CCOEFF_NORMED)
	return [mf.max()]

def highFreqMetrics(P, bins):
	""" High frequency statistics
		
		Calculate statistics of features at higher frequencies
		This is designed to capture false alarms that occur
		at frequencies higher than typical whale calls.

		Also sum accross frequencies to get an average temporal
		profile. Then return statistics of this profile. The
		false alarms have a sharper peak.

		Args:
			P: 2-d numpy array image
			bins: time bins

		Returns:
			A list containing the standard deviation of the
			centroid, the mean of the width, and then 
			the moments of the collapsed slices

	"""
	Q = slidingWindowH(P,inner=7,maxM=50,norm=True)[25:,:]	
	m, n = Q.shape
	cf_ = np.empty(m)
	bw_ = np.empty(m)
	for i in range(m):
		mQ = Q[i,:]
		min_, max_ = mQ.min(), mQ.max()
		mQ = (mQ - min_)/(max_ - min_)
		cf_[i] = np.sum(mQ*bins)/np.sum(mQ)
		bw_[i] = np.sqrt(np.sum(mQ*(bins-cf_[i])*(bins-cf_[i]))/np.sum(mQ))

	mQ = np.sum(Q[14:,:],0)
	min_, max_ = mQ.min(), mQ.max()
	mQ = (mQ - min_)/(max_ - min_)
	cfM_ = np.sum(mQ*bins)/np.sum(mQ)
	bwM_ = np.sqrt(np.sum(mQ*(bins - cfM_)*(bins -cfM_))/np.sum(mQ))

	return [np.std(cf_), np.mean(bw_), cfM_, bwM_]


def templateMetrics(P, tmpl):
	""" Template matching

		Perform template matching for a list of templates

		Args:
			P: 2-d numpy array image
			tmpl: templateManager object

		Returns:
			List of correlations, x and y pixel locations of the max 
	"""
	maxs, xs, ys = [], [], []
	for k in range(tmpl.size):
		mf, y, x  = matchTemplate(P,tmpl.templates[k])
		maxs.append(mf)
		xs.append(x)
		ys.append(y)
	return maxs + xs + ys


def normalize(P):
	""" Converts image to binary

		Args:
			P: 2-d numpy array image

		Returns:
			2-d numpy array image
	"""
	tmpl = P.copy()
	m_, s_ = np.mean(tmpl), np.std(tmpl)
	min_ = tmpl.min()
	tmpl[ tmpl < m_ + 0.5*s_] = min_
	tmpl[ tmpl > min_ ] = 1
	tmpl[tmpl < 0] = 0
	return tmpl

def load_tmpl2(tmpl2_dir):
	""" Load templates
		Args:
			tmpl2_dir: path to CSV files with templates

		Returns:
			List of:
				4 2-d numpy array images
				A list of integers with location of templates in the images
	"""
	f_names = ["TV1.csv", "TV1.csv", "TV1.csv", "TV1.csv", "xloc.csv"]
	TV1, TV0, TH1, TH0, xyloc = [np.loadtxt(tmpl2_dir + x, delimiter=",") for x in f_names]
	return [TV1, TV0, TH1, TH0], xyloc

def template2Metrics(P, tmpl2, xyloc):
	""" Template matching
	Args:
	    P: 2-d numpy array image
	    tmpl2: templates
	    xyloc: A list of integers with location of templates in the images

	Returns:
		list of maximum correlation and location, per each template (4)
	"""
	out = []
	for i in range(4):
		x0, xn, y0, yn = xyloc[i]
		tmpl2i = normalize(tmpl2[i][x0:xn,y0:yn].astype('Float32'))
		mf, y, x  = matchTemplate(P.astype('Float32'),tmpl2i)
		out.append(mf)
		out.append(x)
		out.append(y)
	return out

def diffMetrics(P, tmpl2, maxM):
	""" Compute distance between image and template (both in binary form)
	Args:
	    P: 2-d numpy array image
	    tmpl2: templates
	    maxM: max frequency slice to consider

	Returns:
		One value per template (4)
	"""
	out = []
	for i in range(4):
		tmpl2i = normalize(tmpl2[i][:maxM,:].astype('Float32'))
		Pn = normalize(P.astype('Float32'))
		out.append( np.sqrt( ((Pn - tmpl2i)**2).sum()) )
	return out




