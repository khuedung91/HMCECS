import numpy as np
import rpy2.robjects as robjects

# diagnostic
r = robjects.r
r.library('coda')
EffectiveSize = r['effectiveSize'] # Get function of interest	

def nparray2rmatrix(x):
    """
    Converts a nparray to an r matrix.
    """
    try:
	nr, nc = x.shape
    except ValueError:
	nr = x.shape[0]
	nc = 1
    xvec = robjects.FloatVector(x.transpose().reshape((x.size)))
    xr = robjects.r.matrix(xvec, nrow=nr, ncol=nc)
    return xr
    
def IF(thetaDraws, burnin):
    """
    Compute the computational time which is defined as CT = IF*n, where n is the total number of density evaluations the algorithm has performed and IF is the inefficiency factor
    The function returns CT for each parameter
    """
    pVar = thetaDraws.shape[1]    
    IF= np.zeros(pVar)
    ESS = np.zeros(pVar)
    for j in xrange(pVar):
	draws = thetaDraws[burnin:, j]
	
	# Compute effective samples 
	PosteriorDrawsBetaRFormat = nparray2rmatrix(draws)
	ESS[j] = np.array(EffectiveSize(PosteriorDrawsBetaRFormat))			
	IF[j] = len(draws)/ESS[j]	
	
    
    return  IF, ESS