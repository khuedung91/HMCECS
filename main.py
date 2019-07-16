# main code
# code to implement HMC-ECS for logistic regression
# please have a look at what each line is doing before running the script
# this code is written in python 2.7. 
# If you're using python 3, then change the "from __future__ import" to "import"
from __future__ import division
import os
#-------------------------------------------------#
# load some additional library for plotting
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings('error')
import numdifftools as nd
from scipy.optimize import minimize
from scipy.stats import norm
os.chdir('HamiltonianMonteCarlo\HMCwG\sampleCode') #change to your working directory
#--------------------------#
# load functions
# functions contains all necessary functions to run HMCECS, including the sampler
# IF_code used to compute Inefficiency Factor using package CODA in R. Requires rpy2. 
# If you have problems installing rpy2 then don't import IF_code.py
from functions import*

#from IF_code import*

npar = 10
np.random.seed(1234)   
dataFile = 'data'+str(npar)+ 'Par'

#------------------------------------------------#   
data_ip = np.loadtxt(dataFile +'.txt')
#beta_true = np.loadtxt('betaTrue.txt') 

n= len(data_ip)

#----------------------------------------------------#
# define prior
priorInfo = getPrior('gaussian',npar,scale = 5) #Assume N(0,5^2) for all parameters

# initialize
betaMP = minimize(minuslogpost, np.zeros(npar), args= (data_ip[:,0],data_ip[:,1:],5),method='Newton-CG', jac=minusDerPost,hess = minusHessPost,options={'disp': True})
beta_bar = betaMP['x']

# mass matrix M. default is identity matrix
pCov = np.identity(npar)
beta = np.random.multivariate_normal(beta_bar,1e-4*np.linalg.inv(pCov),1)[0]

# There are quite a numeber of tuning parameters needed for HMCECS
# They need to be specified as input for the main function


burnin = 600
samples = 1000
logFile = 'test.txt'

# 1. Arguments for adaptation.
# Note that when we set M=I then epsilon is small and I don't spend too much time in the first few iterations
# This is obtained by by changing epsilon*L everytime I think M changes a lot. 
# Note that M is updated every time the reference for control variate is updated
# Not necessary when M is fixed or you have the optimal M. 
# The setting are arbitrary- just make sure that we have enough iterations with the desired epsilon*L 
# and the value in the last 2 elements of trajLength is your desired epsilon*L
phaseStart = np.array([   1,100,200, burnin]) #if M is fixed : np.array([   1,burnin])
trajLength =np.array([ 0.1,1.2,1.2 ,1.2 ]) # if M is fixed : np.array([ 1 ,1 ])

adaptInfoSub = {'adapt':True,'alpha': 0.80,'gamma': 0.05, 'kappa': 0.75,'t0':10,'updateM':True,
'phaseStartPt': phaseStart,'tol': 1e-8,'diagM':False,'maxEps':1,'cov':'full','updateRef':True}

# some additional information for the hmc step
hmcInfo = {'eps':0.001,'trajLength':trajLength,'pCov': pCov,'maxSteps':300}

# choose either the perturbed or signed (exact) version
algorithm = 'perturbed'


if (algorithm == 'perturbed'):
    # some additional settings for subsampling part: 
    # subsize: subsample size. Note that in practice you should estimate the variance of the log-likelihood estimator and use that information to set the subsample size
    # biasCorrect: (True/False) for perturbed version only, whether to add the bias correction to likelihood and gradient evaluation
    # updateFreq: how often the reference theta* is updated (eg: every 200 iterations)
    # updateU: (True/False) whether to run the first step of HMCECS or not
    # rho: correlation in the blocking of subsample. rho = 0.99 is equivalent to update 1% of the subsample only
    # order: the order of the Taylor expansion 
    subsampleApprox = {'subsize': int(n*0.01),'biasCorrect':True,'updateFreq':200,'updateU':True,'rho' : 0.99,'order':2}
    output = hmc_within_gibbs(data_ip[:,0],data_ip[:,1:], beta, beta_bar,burnin,samples,hmcInfo,priorInfo,subsampleApprox,adaptInfoSub,logFile)
else:
    # some additional settings for subsampling part: 
    # subsize: subsample "chunk" size. Please refer to paper
    lambda_ = 100
    dhat_mean = 0
    a_ = dhat_mean-lambda_
    mb = 30#
    # note that signed HMCECS are implemented with 2nd order Taylor expansion control variate only
    subsampleExact = {'subsize': mb,'lambda':lambda_,'a':a_,'biasCorrect':True,'updateFreq':200,'updateU':True,'rho' : 0.99,'order':2}
    output = hmc_ecs_Exact(data_ip[:,0],data_ip[:,1:], beta, beta_bar,burnin,samples,hmcInfo,priorInfo,subsampleExact,adaptInfoSub,logFile)

# note that the output is an dictionary
np.save('HMCECS1.npy',output)
#----------------------------------------------------#
# to test- full data HMC

adaptInfoHMC = {'adapt':True,'alpha': 0.80,'gamma': 0.05, 'kappa': 0.75,'t0':10,'updateM':True,
'phaseStartPt': phaseStart,'tol': 1e-8, 'diagM':False,'maxEps':1,'updateFreq':200}

logFile ='test.txt'

hmcFull = hmc(data_ip[:,0],data_ip[:,1:], beta,burnin,samples,0.001,trajLength,300,pCov,priorInfo,adaptInfoHMC,logFile)

np.save('HMCFit.npy',hmcFull)


#for p in xrange(npar):
#    plt.figure(p)
#    plt.plot(hmcFull['par']['theta'][:,p],label = 'hmc')
#    plt.plot(output['par']['theta'][:,p],label = 'hmcecs')
#    plt.legend()

#----------------------------------------------------#
# SGHMC
runSGHMC = False
if (runSGHMC):
    ControlVariate = 2
    m = int(n*0.01)
    epsilon = 0.2
    sghmcPar = {'C':np.identity(npar),'V':np.zeros([npar,npar]),'subsize':m}
    adaptSGHMC = {'updateFreq':200, 'updateM':False,'diagM':False,'adapt':False,'phaseStartPt': [1]}
    
    sghmc_1= sghmcMat(data_ip[:,0],data_ip[:,1:], beta,beta_bar,burnin,samples,epsilon,[1.2],hmcFull['M'],sghmcPar,priorInfo,adaptSGHMC,ControlVariate,logFile,saveTempOutput=False)
