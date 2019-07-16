from __future__ import print_function
from __future__ import division
import numpy as np
import sys
from scipy.stats import uniform
from scipy.stats import multivariate_normal
from scipy.stats import halfcauchy
from scipy.stats import invgamma
from scipy.linalg import sqrtm
import scipy.optimize
import random
import numpy.random as npr
import copy
import time
import scipy.stats as sps
#import mpmath as mp
import scipy.special
import os
import numdifftools as nd
from scipy.special import polygamma, gammaln


""" Sample code for HMCECS
""" 
# this file contains all funtions


np.random.seed(1523)
#precision = 150
#mp.mp.prec += precision
def loglike_ind(y,x,theta):
    
    """this function evaluate the individual log likelihood at individual level
     input is whole data set
    """
    
    npar = np.shape(x)[1]
    tmp = x.dot(theta[:npar])
    if np.any(tmp > 700):
        tmp[np.where(tmp>700)]=700
    
    out = (y.T)*tmp - np.log1p(np.exp(tmp))
    
    if(np.isnan(out).any()==True):
        out[np.where(np.isnan(out))] = np.log(1e-100)
    return out  
    
#---------------------------------------------------#
def loglike_all(y,x,theta):
    """ 
    loglikelihood of whole data set
    """
    npar = np.shape(x)[1]
    ll_ind = loglike_ind(y,x,theta[:npar])
    out = np.sum(ll_ind)
    return out
  
#-----------------------------------------------------------#
def hessianll(x,theta):
    """function that evaluate the second derivative of the likelihood at a value of the data , for 1 data point"""
    
    npar = len(x) # number of real parameter
    out = np.zeros([len(theta),len(theta)])
    out[:npar,:npar] = -np.exp(np.dot(x,theta[:npar]))/(1+np.exp(np.dot(x,theta[:npar])))**2*np.outer(x,x)
    return out
    
def hessianll2(x,theta):
    """function that evaluate the second derivative of the (individual) likelihood at a value of the data """
    npar = np.int(np.shape(x)[1])
    
    tmp = -1/(np.exp(0.5*np.dot(x,theta[:npar]))+np.exp(-0.5*np.dot(x,theta[:npar])))**2
    out = np.zeros([len(x),len(theta),len(theta)])
    out[:,:npar,:npar] = np.array(map(np.multiply,x[:,:,None]*x[:,None,:],tmp))
    
    return out

#------------------------------------------------------------#
def sumHessian(x,theta):
    n = len(x)
    H = 0
    for i in range(0,n):
        H = H + hessianll(x[i],theta)
    return H
    
#-----------------------------------------------------------#
def hessianPrior(theta,family,par1,par2):
    """function that evaluate the second derivative of the prior at a value of the data """

    if(family== 'gaussian'):
        out = -np.linalg.inv(par2)
        
    else:
        print ("prior not defined !")
        out = np.nan
    return out
#---------------------------------------------------#
def devll_ind(y,x,theta):
    
    """
    function that evaluate the first derivative of the log likelihood at a value of the data
    for individual
    """
    
    #npar = np.shape(data)[1]-1L
    npar = len(x)
    tmp = np.dot(x,theta[:npar])
    if (tmp < -700):
       tmp=-700
    out = np.zeros(len(theta))
    out[:npar] = y*x - x*1/(1+np.exp(-tmp))
    return out  
#-------------------------------------------#
def devll(y,x,theta):
    """ for a set of observations
    """
    npar = np.shape(x)[1] #number of 'real' parameters- betas
    tmp = np.dot(x,theta[:npar])
    if (np.any(tmp < -700)==True):
       tmp[np.where(tmp<-700)]=-700
    tmp = 1/(1+np.exp(-tmp)) 
    out = np.zeros([len(y),npar])
    out[:,:npar] = np.multiply(x.T,y-tmp).T   
    #out = y*x - np.multiply(x.T,tmp).T
    return out
#--------------------------------------------------#
def sumDev(y,x,theta):
    n = len(y)
    D = 0
    for i in range(0,n):
        D = D+   devll_ind(y[i],x[i],theta)  
    return D
#---------------------------------------------------#    
def dprior(theta,family,par1,par2):
    """
    log prior 
    
    for a gaussian prior, par1 is mean, par2 is covariance matrix
    """
    if family =='gaussian':
        out = -0.5*np.dot((theta-par1),np.linalg.solve(par2,(theta-par1)))
        #out = np.log(max(1e-300,multivariate_normal.pdf(theta,par1,par2)))
    else:
        print ('prior not defined')
        out = np.nan
    return out
    
#------------------------------------------------------#
def gradPrior(theta,prior,npar,priorPar1 = 0, priorPar2  =0):
    
    if prior =='gaussian':
        gradPrior = - np.linalg.solve(priorPar2,theta-priorPar1)
    else:
        print ('prior not defined')
        gradPrior = np.nan
    return gradPrior
#---------------------------------------------------#
def gradU(y,x,theta,prior='uniform',priorPar1=0,priorPar2=0):
    
    """
    function that evaluate the gradient of minus the log posterior 
    """
    npar = np.shape(x)[1]
    const = 1/(1+np.exp(-np.dot(x,theta[:npar])))
    dev_ind = y.reshape(-1,1)*x - const.reshape(-1,1)*x
    gradll= np.zeros(len(theta))
    gradll[:npar] = -np.sum(dev_ind,axis =0) #minus gradient of log likelihood
    gPrior = gradPrior(theta,prior,npar,priorPar1,priorPar2)
    out = gradll  - gPrior
    return out

#----------------------------------------------------#
def potential(y,x,theta,par1,par2,family):
    """ potential energy"""
    U = -(loglike_all(y,x,theta)+dprior(theta,family,par1,par2))
    return U
    
#----------------------------------------------------------#
def kinetic(p,M):
    # technically just minus the log of multivariate normal density with mean 0 and covariance M, without the normalizing constant
    # may be numpy library is faster but just use the formular first
   
    K = 0.5*p.dot(np.linalg.solve(M,p))
    return K
    
#----------------------------------------------------------#
def hmc(y,x, theta,burnin,samples,eps,trajLength,maxSteps,M,priorArgs,adaptArgs,logFile,saveTempOutput=False):
    """ IMPLEMENTATION OF HMC WITH ADAPTIVE WARMUP
        M to be updated using the output from the previous iterations
        eps to be updated continously during burnin
        
    """
    # eps is stepsize (starting value, to be update if burnin>0)
    # L is number of steps (fix)
    # M is covariance matrix of p (to be update if burnin > 0)
    
    # burnin = 0 implies no update at all
    
    #n = len(data)
    niter = burnin + samples
    npar = len(theta)
    nbetas = np.shape(x)[1]
    theta_keep = np.zeros([niter,npar])
    theta_propose = np.zeros([niter,npar])
    acc_rate = np.zeros(niter)
    L_keep = np.zeros(niter)
    L = int(trajLength[0]/eps)
    timePerIter = np.zeros(niter)
    Hdiff = np.zeros(niter)
    # arguments for prior
    pfamily = priorArgs['family']
    priorPar1 = priorArgs['par1']
    priorPar2 = priorArgs['par2']
    
    meanp = np.zeros(len(theta))
    #-------------------------------#
    updateFreq = adaptArgs['updateFreq']
    alpha = adaptArgs['alpha'] #desired acceptance rate
    gamma =adaptArgs['gamma']
    kappa = adaptArgs['kappa']
    updateM = adaptArgs['updateM']
    t0 = adaptArgs['t0']
    Hbar = 0 #Hbar is a sumstat ~ different between the desired acceptance rate and the mean acceptance rate upto time t
    mu = np.log(10*eps)
    logEps = np.log(eps)
    logEpsBar = 0
    #-----------------------------------#
    #
    if(adaptArgs['adapt']==True and burnin>0):
        eps_keep =np.zeros(niter)
        eps_keep[0] = eps
        diagM = adaptArgs['diagM']
        phaseStartPt = adaptArgs['phaseStartPt']
        if(len(phaseStartPt) != len(trajLength)):
            print('phaseEndPt must have same length as trajLength')
            
        phase = 0
    else:
        eps_keep = eps
    #-------------------------------#
    try:
        for i in range(0,niter):
            progress = i*100/niter
            if np.mod(progress,5)==0:
                print(str(progress)+ "% ",end= "")
            if (np.mod(progress,10)==0 and i>0):
                msg = str(progress) + "% ; nsteps now is: " + str(L) + "; mean acc is: " + str(np.mean(acc_rate[:i]))
                print(msg)
                lf = open(logFile,"a")
                lf.write(msg+ '\n')
                lf.close()
                if(saveTempOutput):
                    part = int(progress*0.1)
                    temp = {'par':theta_keep[:i],'eps':eps_keep,'M':M}
                    np.save('output/temp'+ str(part) + '.npy',temp)
                        
            startT = time.time() 
            
               
            p = np.random.multivariate_normal(meanp,M,1)[0]
            thetacurrent = theta
            Hcurrent = -(loglike_all(y,x,theta[:nbetas])+dprior(theta,pfamily,priorPar1,priorPar2))+ kinetic(p,M)
            
            L_keep[i] = L
            # move p by half a step
            p = p-0.5*eps*gradU(y,x,theta,pfamily,priorPar1,priorPar2)
            for s in range(0,L):
                # move position
                theta = theta+ eps*np.linalg.solve(M,p)
                # move momentum
                if s<(L-1):
                    p = p - eps*gradU(y,x,theta,pfamily,priorPar1,priorPar2)
                else:
                    p = p - 0.5*eps*gradU(y,x,theta,pfamily,priorPar1,priorPar2)
                
            # negate p
            p = -p
            theta_propose[i] = theta
            H = -(loglike_all(y,x,theta[:nbetas])+dprior(theta,pfamily,priorPar1,priorPar2))+ kinetic(p,M)
        
            
            reject = False
            Hdiff[i] = Hcurrent - H
            accrate = np.exp(min([0,(Hcurrent-H)]))#
            reject = (np.random.uniform(0,1,1)> accrate)
            if reject==True:
                theta = thetacurrent
                H = Hcurrent
            stopT = time.time()
            timePerIter[i] = stopT- startT
            
            #------------------------------------------------#
            # adjust eps and M if burnin > 0
            # don't evaluate the time getting new eps for adaptive tunning as it's small
            if(burnin >0):
                t = i+ 1
                if(adaptArgs['adapt']==True):
                    if(updateM and np.mod(i+1,updateFreq)==0 ):
                        startT = time.time()
                        if(diagM ==True):
                            var_theta = np.var(theta_keep[int(i/2):(i+1),:],axis = 0) #take half the iteration as the first iters might not be informative
                            M= np.diag(1/var_theta)
                        else:
                            
                            thetaRef = np.mean(theta_keep[int(0.5*i):i,:],axis =0)
                            
                            Hprior = hessianPrior(thetaRef,pfamily,priorPar1,priorPar2)
                            M = -sumHessian(x, thetaRef) - Hprior
                        stopT = time.time()
                        timePerIter[i] = timePerIter[i] + stopT- startT  
                    
                    if(i< burnin):
                        Hbar = (1-1/(t+t0))*Hbar + 1/(t+t0)*(alpha-accrate)
                        logEps = mu - np.sqrt(t)/gamma*Hbar
                        logEpsBar = t**(-kappa)*logEps + (1-t**(-kappa))*logEpsBar
                                                                                                        
                        if(phase < len(phaseStartPt)):
                            if((i+1)==phaseStartPt[phase]):
                                currentTrajLength = trajLength[phase]
                                phase +=1 #next phase is phase 1
                            # reset
                            #if(phase < len(phaseStartPt)):
                            #    mu = np.log(10*eps)
                            #    logEps = np.log(eps)
                            #    logEpsBar = 0
                            #    Hbar = 0 
                        eps = np.min([adaptArgs['maxEps'],np.exp(logEps),currentTrajLength])
                        eps_keep[t] = eps
                        L = min(maxSteps,int(round(currentTrajLength/eps,0)))   
                        if (L==0):
                            print('number of steps reaches 0!')
                            L +=1     
                        if(t==burnin) :
                            # t == burnin
                            eps = np.min([np.exp(logEpsBar),adaptArgs['maxEps'],currentTrajLength])
                            eps_keep[t:] = eps
                            L = min(maxSteps,int(round(trajLength[-1]/eps,0)))
                            L_fix = max(1,L)
                    
                    else:
                        L = L_fix
                else:
                    if t < burnin :
                        L = int(round(trajLength[0]/eps,0))
                    else:
                        L = int(round(trajLength[1]/eps,0))
                                                                
                
            theta_keep[i]= theta
            acc_rate[i] = accrate
    except Warning,w:
        print (str(w))
    except TypeError,e:
        print('type error' + str(e))
    except ValueError,e:
        print('value error'+ str(e))
    except IndexError,e:
       print('Index error'+ str(e))
    except (KeyboardInterrupt, SystemExit):
        print('Bye')
    lf = open(logFile,"a")
    lf.write('Run completed successfully')
    lf.close()
    
    finalpar = {'theta':theta_keep}
    currentSet = {'theta':theta,'iter':i}
    return {'par':finalpar,'acc': acc_rate,'eps':eps_keep,'nsteps':L_keep,'M':M,'Hbar':Hbar,'runTime':timePerIter,
    'Hdiff': Hdiff,'current':currentSet,'proposal':theta_propose,'args':adaptArgs}
      
#--------------------------------------------------------#
def proxy_ind(y,x,theta,thetaref,d1,d2=0,order = 2):
    
    llref = loglike_ind(y,x,thetaref)
    if(order==0):
        q = llref
    else:
        q = llref + np.dot(d1,(theta-thetaref))# 
        if (order ==2):
            q = q + 0.5*(theta-thetaref).dot(d2).dot(theta-thetaref)
    return q
#---------------------------------------------------------#
def proxy_ind1stOrder(y,x,theta,thetaref,d1):
    
    llref = loglike_ind(y,x,thetaref)
    q = llref + np.dot(d1,(theta-thetaref)) 
    return q
    
#--------------------------------------------------------#
def diff_ind(y_u,x_u,theta,thetaref,d1_u,d2_u=0,order = 2):
    
    # u is the vector of index of obs included
    #evaluate d_i = l_i- q_i
    lltrue = loglike_ind(y_u,x_u,theta)
    
    proxy = proxy_ind(y_u,x_u,theta,thetaref,d1_u,d2_u,order)
    diff = np.array(lltrue) - np.array(proxy)
    return diff
    
#--------------------------------------------------------#
def proxy_sum(theta,thetaref,llref,g,H=0,order=2):
    if(order ==0):
        proxySum =llref
    else:
        proxySum = llref + g.dot(theta-thetaref) #
        if (order==2):
            proxySum = proxySum + 0.5*(theta-thetaref).dot(H).dot(theta-thetaref)
    return proxySum

#--------------------------------------------------------#
def loglike_est(theta,thetaref,llref,g,H,diff_i,n,order = 2):
    """ log likelihood estimate. Require computing d_i beforehand """
    m = len(diff_i)
    sumProxy = proxy_sum(theta,thetaref,llref,g,H,order)
    out = sumProxy + n/m*np.sum(diff_i)
    
    return out

#---------------------------------------------------------#
def sumGradU_est(theta,thetaRef,dev1_sum,dev2_sum=0,order =2):
    """ SUM OF ESTIMATED GRADIENT FOR ALL N """
    if(order==0):
        sumgrad = 0
    else:
        sumgrad = dev1_sum #
        if(order==2):
            sumgrad = sumgrad + dev2_sum.dot(theta-thetaRef)
    return sumgrad
#------------------------------------------------------------#
def init_u(mb,n,algorithm,lambda_=50,rho = 0.99):
    """
    Note: n is the population size, m is the size of the subsample, G is the number of blocks
    """
    G = np.round(1/(1-rho),0)
    if algorithm =='Approx':
        # Update one of the blocks:
        uCurr = npr.randint(0, n, mb) 
        
        # Divide the random variates into blocks
        groupindicators = np.hstack((np.repeat(np.arange(G-1), mb/G), np.repeat(G-1, mb - len(np.repeat(np.arange(G-1), mb/G)))))	
    
        return uCurr, groupindicators
    elif algorithm == 'Exact':
        """ Initialize u for the product poisson estimator
            uCurr has fixed length(lambda)
            each element of uCurr has a random length base on X_l ~ Poisson(1)
            each component of each element of uCurr has fixed length m
        """
        uCurr = [[npr.choice(np.arange(n),mb) for item in xrange(sps.poisson.rvs(1))] for item in xrange(lambda_)]
        Gc = np.sum([len(item) for item in uCurr]) # total number of u's used. The cost is Gc*m
        kappa = int(round(lambda_/G,0))
        
        return uCurr, Gc, kappa 
    else:
        print('Invalid algorithm')


def uProp_given_uCurr(mb, n, uCurr, algorithm, lambda_, groupindicators= 0,kappa = 1):
    """
    Propoposes u given uCurr 
    """
    if algorithm =='Approx':
    # Update one of the blocks:
        G = max(groupindicators)+1
        toUpdate = npr.randint(0, G, 1)[0]
        uProp = copy.copy(uCurr)
        update = (groupindicators == toUpdate)
        uProp[update] = npr.randint(0, n, np.sum(update)) 
        
        return uProp,np.where(update)
    elif algorithm == 'Exact':
        """ Choose one block of uCurr and update all the u there"""
        uProp = copy.copy(uCurr)
        
        BlockToChange = npr.choice(lambda_, kappa, replace = False)
        for b in BlockToChange:
            uProp[b] = [npr.choice(np.arange(n),mb) for item in xrange(sps.poisson.rvs(1))]
        Gp = np.sum([len(item) for item in uProp]) #cost at proposed is Gp*m
        return uProp, Gp, BlockToChange
        
    else:
        print('Invalid algorithm')
#--------------------------------------------------------------#
def component_xi(y_xi,x_xi,theta,thetaref,dev1_xi,dev2_xi,llref,a,lambda_,n):
    """ use this to calculate one xi_l only """
    
    # data is a list of matrix of subdata (length = Chi_l)
    # something like data_subFull= [[data[subset] for subset in item] for item in u] which has length lambda
    # then for each xi use data_subFull[l], which is of length xi, and is a list of matrices of size m*p
    # similarly for dev1_xi and dev2_xi
    Chi_l = len(y_xi)
    diff_h = np.zeros(Chi_l)
    if Chi_l ==0 :
        xi_l = np.exp(1+a/lambda_)
        sigma2_dhat = 0
    else:
        m = np.shape(x_xi[0])[0]
        var_set = np.zeros(Chi_l)
        for h in range(0,Chi_l):
            diffs = diff_ind(y_xi[h],x_xi[h],theta,thetaref,dev1_xi[h],dev2_xi[h])
            diff_h[h] =n/m*np.sum(diffs) #d_hat_m
            
           
        xi_l = np.exp(1+a/lambda_)*np.prod((diff_h-a)/lambda_)   
        sigma2_dhat = np.mean(var_set)        
    return xi_l,sigma2_dhat

#------------------------------------------------------------------------#s
def sumStats(theta_keep):
    mean = np.mean(theta_keep,axis = 0)
    sd = np.std(theta_keep,axis = 0)
    return{'mean': mean,'std':sd}
    
#--------------------------------------------------------------------------#
def Var_abs_LogL_hat(m, lambda_, gamma, sum_trunc = 100):
    """
    sum_trunc: number of terms to include before truncating the sum
    """
    mean_pois = m*lambda_**2/(2*gamma)
    #sum_trunc = sum_trunc*np.ones(mean_pois.shape).astype(int)
    upper = np.int(np.max([sum_trunc, sps.poisson.ppf(0.99999999, mean_pois)])) # np.maximum.reduce([sum_trunc, sps.poisson.ppf(0.99999999, mean_pois)])	
    JPois = np.arange(upper)
    nu2 = 0.25*(np.sum(polygamma(1, 0.5 + JPois)*sps.poisson.pmf(JPois, mean_pois)) + np.sum((polygamma(0, 0.5 + JPois) - np.sum(polygamma(0, 0.5 + JPois)*sps.poisson.pmf(JPois, mean_pois)))**2*sps.poisson.pmf(JPois, mean_pois)))
    eta = np.log(np.sqrt(gamma/(m*lambda_**2))) + 0.5*(np.log(2) + np.sum(polygamma(0, 0.5 + JPois)*sps.poisson.pmf(JPois, mean_pois)))
    
    return lambda_*(nu2 + eta**2)
      
#-----------------------------------------------------------------------------#
def dhat_block(y_u,x_u,theta,thetaref,d1,d2,n,gradient = False,order = 2):
    # compute d_hat_m in block (h,l)
	# data_u is data of block h,l
    m = len(x_u)
    if (m>0):
        
        lltrue = loglike_all(y_u,x_u,theta)
        llref = loglike_all(y_u,x_u,thetaref) # sum of l_i in sub-block h,l
        if(order ==2):
            dhat_m_hl = n/m*(lltrue - llref - np.dot(d1,(theta-thetaref)) - 0.5*(theta-thetaref).dot(d2).dot(theta-thetaref))
        else:
            if(order==1):
                dhat_m_hl = n/m*(lltrue - llref - np.dot(d1,(theta-thetaref)))
            else:
                dhat_m_hl = n/m*(lltrue - llref)
        #q = llref + np.dot(d1,(theta-thetaref)) + 0.5*(theta-thetaref).dot(d2).dot(theta-thetaref)
        if(gradient):
	   dev1_component = np.sum(devll(y_u,x_u,theta),axis = 0)
	   grad_logdhat_m = dev1_component-d1  # note that this is the part of the gradient without the denominator
	   if(order ==2):
	      grad_logdhat_m = grad_logdhat_m - d2.dot(theta-thetaref) 
	   return grad_logdhat_m,dhat_m_hl    
        else:
	   return dhat_m_hl
    else:
        dhat_m_hl = 0
	if(gradient):
	   grad_logdhat_m = 0# note that this is the part of the gradient without the denominator
	   return grad_logdhat_m,dhat_m_hl    
        else:
	   return dhat_m_hl
        
#---------------------------------------------------------------------------------#

def loglike_estPoissonWithGradient(y_sub,x_sub,theta,thetaref,n,subsamplingDict,getLoglike = True,getGradient=True):
    # data is a list of matrix of subdata (length = Chi_l)
    # something like data_subFull= [[data[subset] for subset in item] for item in u] which has length lambda
    # then for each xi use data_subFull[l], which is of length xi, and is a list of matrices of size m*p
    # similarly for dev1_xi and dev2_xi
    # output gradient of minus log-likelihood as well
    a = subsamplingDict['a']
    lambda_ = subsamplingDict['lambda']
        
    dhat_m = [[dhat_block(y_sub_item,x_sub_item,theta,thetaref,sub_item2,sub_item3,n,order = subsamplingDict['order']) for y_sub_item, x_sub_item, sub_item2,sub_item3 in zip(y_item,x_item,item2,item3)] 
    if (len(y_item)>0) else [] for y_item,x_item,item2,item3 in zip(y_sub,x_sub,subsamplingDict['dev1_sumComponent'],subsamplingDict['dev2_sumComponent'])]
                
    if(getLoglike):
        component_xi = [(np.asarray(item)-a)/lambda_ if (len(item)>0) else [1] for item in dhat_m]
        logprods = [np.log(np.abs(item))  for item in component_xi]
        signL = np.prod([np.prod(np.sign(xi)) for xi  in component_xi])
                # report log likelihood 
        l_hat = subsamplingDict['sumProxy'] + a + lambda_ +  np.sum([np.sum(item) for item in logprods]) #log(|Lhat|)
                #        
        #----------------------------#
        # get variance
        """ This is pretty costly to compute and not neccessary so I skipped it"""
        #dhat = np.mean([np.mean(item) for item in dhat_m])
        #lenItem = np.array([len(item) for item in dhat_m])
        #nonEmpty = np.where(lenItem>1)[0][0]#[1:10]
        
        #newlist = [item for sublist in dhat_m for item in sublist]
        #gamma =subsamplingDict['subsize']*np.var(newlist)
        #sigma2_LL = Var_abs_LogL_hat(subsamplingDict['subsize'], lambda_, gamma, sum_trunc = 100)
        sigma2_LL= 0
    else:
        l_hat = 0
        signL = 1
        sigma2_LL  = 0
    #-------------------------------------------------------------------#
    # calculating gradient
    
    if(getGradient):
        
        dev1_component = [[np.sum(devll(y_subset,x_subset,theta),axis = 0) for y_subset,x_subset in zip(y_item,x_item)] for y_item,x_item in zip(y_sub,x_sub)]
        
        sumGrad =  sumGradU_est(theta,thetaref,subsamplingDict['sumDev1'],subsamplingDict['sumDev2'],order = subsamplingDict['order']) # grad(q)
        grad_logprods =  [np.sum([(sub_item1-sub_item2- sub_item3.dot(theta-thetaref))/(sub_item4-a) for sub_item1,sub_item2,sub_item3,sub_item4 in zip(item1,item2,item3,item4)],axis =0) 
        if (len(item1)>0) else np.zeros(len(theta)) 
        for item1,item2,item3,item4 in zip(dev1_component,subsamplingDict['dev1_sumComponent'],subsamplingDict['dev2_sumComponent'],dhat_m)]
 
        grad_minusloglike = -(sumGrad + n/subsamplingDict['subsize']*np.sum(grad_logprods,axis = 0))
    else:
        grad_minusloglike = 0
    
    return grad_minusloglike,l_hat,signL,sigma2_LL
        
#-----------------------------------------------------------------------------#
def minusloglike_estPoisson(theta,y,x,u,thetaref,n,subsamplingDict):
    subsamplingDict['sumProxy'] = proxy_sum(theta,thetaref,subsamplingDict['llRef'],subsamplingDict['sumDev1'],subsamplingDict['sumDev2'])
    a = subsamplingDict['a']
    lambda_ = subsamplingDict['lambda']
        
    dhat_m = [[dhat_block(y[sub_item1],x[sub_item1],theta,thetaref,sub_item2,sub_item3,n,order = subsamplingDict['order']) for sub_item1,sub_item2,sub_item3 in zip(item1,item2,item3)] if (len(item1)>0) else [] for item1,item2,item3 in zip(u,subsamplingDict['dev1_sumComponent'],subsamplingDict['dev2_sumComponent'])]

    component_xi = [(np.asarray(item)-a)/lambda_ if (len(item)>0) else [1] for item in dhat_m]
    logprods = [np.log(np.abs(item))  for item in component_xi]
   # signL = np.prod([np.prod(np.sign(xi)) for xi  in component_xi])
            # report log likelihood 
    l_hat = subsamplingDict['sumProxy'] + a + lambda_ +  np.sum([np.sum(item) for item in logprods]) #log(|Lhat|)
    return -l_hat 

gradminusLoglike=nd.Gradient(minusloglike_estPoisson)
hessian_minusLoglike = nd.Hessian(minusloglike_estPoisson)
hessian_minusLoglikeDiag = nd.Hessdiag(minusloglike_estPoisson)#
#----------------------------------------------------------------------------#
def hmc_ecs_Exact(y,x, theta, thetaRef,burnin,samples,hmcArgs,priorArgs,subsampleArgs,adaptArgs,logFile,saveTempOutput=False):
    """ 
        Implement with adaptive eps. Fix trajectory length (L*eps)
        
        I haven't test 1st order exact hmcecs
    """
    n = len(x)
    
    niter = burnin + samples
    npar = len(theta)
    #---------------------------#
    # arguments for prior
    pfamily = priorArgs['family']
    priorPar1 = priorArgs['par1']
    priorPar2 = priorArgs['par2']
    #---------------------------------#
    # HMC argument
    meanp = np.zeros(npar)
    eps = hmcArgs['eps']
    trajLength = hmcArgs['trajLength']
    L = int(round(trajLength[0]/eps,0))
    M = hmcArgs['pCov']
    maxSteps = hmcArgs['maxSteps']
    Mhalf = np.linalg.cholesky(M)
    #--------------------------------------#
    # subsampling arguments
    m = subsampleArgs['subsize']
    a_ = subsampleArgs['a']
    lambda_ = subsampleArgs['lambda']
    rho = subsampleArgs['rho']
    nblocks = 1/(1-rho)
    updateFreq = subsampleArgs['updateFreq']
    updateU = subsampleArgs['updateU']
    cvorder = subsampleArgs['order']
    #-------------------------------------------#
    # adapt arguments
    updateM = adaptArgs['updateM']
    
    
    if(adaptArgs['adapt']==True and (burnin >0)):
        Hbar = 0 #Hbar is a sumstat ~ different between the desired acceptance rate and the mean acceptance rate upto time t
        alpha = adaptArgs['alpha'] #desired acceptance rate
        gamma =adaptArgs['gamma']
        kappa = adaptArgs['kappa']
        t0 = adaptArgs['t0']             
        mu = np.log(10*eps)
        logEps = np.log(eps)
        logEpsBar = 0
        eps_keep = np.zeros(niter)
        
       
    else:
        eps_keep = eps
    phaseStartPt = adaptArgs['phaseStartPt']
    if(len(phaseStartPt) != len(trajLength)):
        print('phaseEndPt must have same length as trajLength')    
    phase = 0
    currentTrajLength = trajLength[phase]
    phase = 1
    #------------------------------------------------#
    # allocate memory for outcome
    theta_keep= np.zeros([niter,npar])
    sigmaHat_keep  = np.zeros([niter,2])
    acc_rate = np.zeros([niter,2])
    L_keep = np.zeros(niter)
    acc_prob = np.zeros([niter,2])
    timePerIter= np.zeros([niter,2])
    signL = np.zeros([niter,2])
    lambda_use = np.zeros(niter)
    #-----------------------------------------------#
    # initialize
    u_m, Gc, unitsPerBlock = init_u(m,n,'Exact',lambda_,rho)
    u0 = copy.deepcopy(u_m)
    
    y_sub = [[y[subset] for subset in item] for item in u_m]
    x_sub = [[x[subset] for subset in item] for item in u_m]
    
    subsampleArgs['sumDev1'] = sumDev(y,x,thetaRef)
    subsampleArgs['sumDev2'] = sumHessian(x, thetaRef) 
    subsampleArgs['dev1_sumComponent'] = [[np.sum(devll(y_subitem,x_subitem,thetaRef),axis = 0) for y_subitem,x_subitem in zip(y_item,x_item)] for y_item,x_item in zip(y_sub,x_sub)]
    subsampleArgs['dev2_sumComponent'] =[[np.sum(hessianll2(x_subitem,thetaRef),axis = 0) for x_subitem in item] for item in x_sub]
    subsampleArgs['llRef'] = loglike_all(y,x,thetaRef)
    subsampleArgs['sumProxy'] = proxy_sum(theta,thetaRef,subsampleArgs['llRef'] ,subsampleArgs['sumDev1'],subsampleArgs['sumDev2'])
    
    
    _,loglikeEst, signl,sigma2LL = loglike_estPoissonWithGradient(y_sub,x_sub,theta,thetaRef,n,subsampleArgs,getLoglike=True,getGradient=False)
    potentialEst = -loglikeEst - dprior(theta,pfamily,priorPar1,priorPar2)
    
    print("Start HMCECS")
    try:
        for i in xrange(niter):
            
            lambda_use[i]= lambda_
            progress = i*100/niter
            
            if (np.mod(progress,10)==0 and i>0):
                msg = str(progress) + "% ; nsteps now: " + str(L) + "; mean acc: " + str(np.mean(acc_prob[:i,1]))+ "; mean acc_u: " + str(np.mean(acc_prob[:i,0]))
                print(msg)
                lf = open(logFile,"a")
                lf.write(msg+ '\n')
                lf.close()
                if(saveTempOutput):
                    part = int(progress*0.1)
                    temp = {'par':theta_keep[:i],'eps':eps_keep,'M':M,'signL':signL[:i]}
                    np.save('output/temp'+ str(part) + '.npy',temp)
            #-------------------------------------#
            if(updateU):
                startT = time.time()
                potentialCurrent = potentialEst
                signl_current = signl
                ucurrent = copy.copy(u_m)
                Gcurrent = Gc
                u_m, Gc, toUpdate = uProp_given_uCurr(m,n,ucurrent,'Exact',lambda_,kappa = unitsPerBlock)
                
                #------------------------------#
                # next update the subsampling dictionary
                for b in toUpdate:
                    
                    y_sub[b] = [y[subset] for subset in u_m[b]] 
                    x_sub[b] = [x[subset] for subset in u_m[b]] 
                    subsampleArgs['dev1_sumComponent'][b] = [np.sum(devll(y_subitem,x_subitem,thetaRef),axis = 0) for y_subitem,x_subitem in zip(y_sub[b],x_sub[b])]
                    subsampleArgs['dev2_sumComponent'][b] = [np.sum(hessianll2(x_subitem,thetaRef),axis = 0) for x_subitem in x_sub[b]]
                    
                    
                # accept/reject
                # 1st compute new (log)likelihood estimate
                _,loglikeEst, signl,sigma2LL = loglike_estPoissonWithGradient(y_sub,x_sub,theta,thetaRef,n,subsampleArgs,getLoglike=True,getGradient=False)
                potentialEst = -loglikeEst - dprior(theta,pfamily,priorPar1,priorPar2)
                
                la_u = np.min([0,-potentialEst+potentialCurrent])
                reject_u = np.log(np.random.uniform(0,1,1))>la_u
                acc_rate[i,0] = 1-reject_u
                acc_prob[i,0] = np.exp(la_u)
                
                # if reject
                if reject_u:
                    potentialEst = potentialCurrent
                    u_m = copy.copy(ucurrent)
                    Gc= Gcurrent
                    signl = signl_current
                    # update the subset again 
                    for b in toUpdate:
                        y_sub[b] = [y[subset] for subset in u_m[b]] 
                        x_sub[b] = [x[subset] for subset in u_m[b]] 
                        subsampleArgs['dev1_sumComponent'][b] = [np.sum(devll(y_subitem,x_subitem,thetaRef),axis = 0) for y_subitem,x_subitem in zip(y_sub[b],x_sub[b])]
                        subsampleArgs['dev2_sumComponent'][b] = [np.sum(hessianll2(x_subitem,thetaRef),axis = 0) for x_subitem in x_sub[b]]
                        
                signL[i,0] = signl
                sigmaHat_keep[i,0] = sigma2LL
                stopT = time.time()
                timePerIter[i,0] = stopT-startT
            
            #---------------------------------#
            # step 2: update theta/u
            startT = time.time()
            #p = np.random.multivariate_normal(meanp,M,1)[0]
            p = Mhalf.dot(np.random.multivariate_normal(meanp,np.identity(len(theta)),1)[0])#np.random.multivariate_normal(meanp,M,1)[0]
            
            thetacurrent = theta
            potentialCurrent = potentialEst
            signl_current = signl
            Hcurrent = potentialCurrent + kinetic(p,M)
            L_keep[i] = L
                
            # first move half a step
            gradEst = loglike_estPoissonWithGradient(y_sub,x_sub,theta,thetaRef,n,subsampleArgs,getLoglike=False,getGradient=True)[0] - gradPrior(theta,pfamily,npar,priorPar1,priorPar2)
            p = p-0.5*eps*gradEst  
            for s in xrange(L):
                theta = theta + eps*np.linalg.solve(M,p)
                # update dictionary
                subsampleArgs['sumProxy'] = proxy_sum(theta,thetaRef,subsampleArgs['llRef'],subsampleArgs['sumDev1'],subsampleArgs['sumDev2'])
                if s< (L-1):
                    gradEst = loglike_estPoissonWithGradient(y_sub,x_sub,theta,thetaRef,n,subsampleArgs,getLoglike=False,getGradient=True)[0] - gradPrior(theta,pfamily,npar,priorPar1,priorPar2)
                    p = p-eps*gradEst  
                else:
                    gradLL,llEstPropose, signl,sigma2LL = loglike_estPoissonWithGradient(y_sub,x_sub,theta,thetaRef,n,subsampleArgs,getLoglike=True,getGradient=True)
                    gradEst = gradLL - gradPrior(theta,pfamily,npar,priorPar1,priorPar2)
                    p = p- 0.5*eps*gradEst
                
            p = -p
            potentialEst = -llEstPropose - dprior(theta,pfamily,priorPar1,priorPar2)
            H= potentialEst + kinetic(p,M)
            la = np.min([0,-H+Hcurrent])
            reject = np.log(np.random.uniform(0,1,1)) >la
            acc_rate[i,1] = 1-reject
            acc_prob[i,1] = np.exp(la)
            
            # if reject
            if reject:
                theta = thetacurrent
                potentialEst = potentialCurrent
                signl = signl_current
                subsampleArgs['sumProxy'] = proxy_sum(theta,thetaRef,subsampleArgs['llRef'],subsampleArgs['sumDev1'],subsampleArgs['sumDev2'])
                
            signL[i,1] = signl
            theta_keep[i]  = theta
            sigmaHat_keep[i,1] = sigma2LL
            stopT = time.time()
            timePerIter[i,1] = stopT - startT
            
            
            #-----------------------------------#
            # adjust eps on the fly
            
            if((burnin>0) and (adaptArgs['adapt']==True)):
                t = i+1
                if((t<= burnin)): #(t>phaseStartPt[1])
                    Hbar = (1-1/(t+t0))*Hbar + 1/(t+t0)*(alpha-np.exp(la))
                    logEps = mu - np.sqrt(t)/gamma*Hbar
                    logEpsBar = t**(-kappa)*logEps + (1-t**(-kappa))*logEpsBar
                    eps = np.min([np.exp(logEps),adaptArgs['maxEps']])
                    if (eps < 0.00001):
                        print('epsilon is getting small!: ' + str(eps))
                        if(eps < 0.00001):
                            print('epsilon is too small!')
                            break
                    # omitted the part where adaption parameters were reset when trajectory changed
                    eps_keep[t] = eps
                    
                
                if(t==burnin):
                    eps = np.min([np.exp(logEpsBar),adaptArgs['maxEps']])
                    eps_keep[i:] = eps
           #------------------------------------------#
            # change trajectory length
            if (phase < len(phaseStartPt)):
                if((i+1)==phaseStartPt[phase] ):
                    currentTrajLength = trajLength[phase]
                    phase +=1 #next phase is phase 1
            L = np.min([maxSteps,int(round(currentTrajLength/eps,0))])
            if(L==0):
                    print('number of steps reached 0 !')
                    L +=1  
            #------------------------------------------#
            # UPDATE REF AND EVERYTHING 
            if( (np.mod(i+1,updateFreq)==0) and (i<burnin) and (adaptArgs['updateRef']==True)):
                if ((i+1) == updateFreq):
                    thetaRef = np.mean(theta_keep[int(0.5*i):i,:],axis =0)
                    
                else:
                    thetaRef = np.mean(theta_keep[int(0.7*i):i,:],axis =0)
                    
                
                subsampleArgs['sumDev1'] = sumDev(y,x,thetaRef)
                subsampleArgs['sumDev2'] = sumHessian(x, thetaRef) 
                subsampleArgs['dev1_sumComponent'] = [[np.sum(devll(y_subitem,x_subitem,thetaRef),axis = 0) for y_subitem,x_subitem in zip(y_item,x_item)] for y_item,x_item in zip(y_sub,x_sub)]
                subsampleArgs['dev2_sumComponent'] =[[np.sum(hessianll2(x_subitem,thetaRef),axis = 0) for x_subitem in item] for item in x_sub]
    
                subsampleArgs['llRef'] = loglike_all(y,x,thetaRef)
                subsampleArgs['sumProxy'] = proxy_sum(theta,thetaRef,subsampleArgs['llRef'] ,subsampleArgs['sumDev1'],subsampleArgs['sumDev2'])
                _,loglikeEst, signl,sigma2LL = loglike_estPoissonWithGradient(y_sub,x_sub,theta,thetaRef,n,subsampleArgs,getLoglike=True,getGradient=False)
                potentialEst = -loglikeEst - dprior(theta,pfamily,priorPar1,priorPar2)
                #--------------------------------------#
                # updating M
                if(updateM):
                    # only update M if the number of negative sign is not to large
                    if (np.sum(signL[int(0.5*i):i,1]<1)/i < 0.2):
                        if(adaptArgs['cov']=='sub'):
                            M = hessian_minusLoglike(thetaRef,y,x,u_m,thetaRef,n,subsampleArgs)- hessianPrior(thetaRef,pfamily,priorPar1,priorPar2)
                        else:
                            M = -(subsampleArgs['sumDev2']) - hessianPrior(thetaRef,pfamily,priorPar1,priorPar2)
                            #M = M*1/(m)
                        

                        if(adaptArgs['diagM']==True):
                            M = np.diag(np.diag(M))
                    if(np.linalg.cond(M)>10**7):
                            print('ill-conditioned covariance matrix!')
                            var_theta = np.var(theta_keep[int(i/2):i,:],axis = 0)
                            M= np.diag(1/var_theta)
                            
                    Mhalf = np.linalg.cholesky(M)
                    
                      
    except Warning,w:
        print (str(w))
    except TypeError,e:
        print('type error' + str(e))
    except ValueError,e:
        print('value error'+ str(e))
    except IndexError,e:
        print('Index error'+ str(e))
    except (KeyboardInterrupt, SystemExit):
        print('Bye')
    lf = open(logFile,"a")
    lf.write('Run completed successfully')
    lf.close()
    
    finalpar = {'theta':theta_keep}
    currentSet = {'theta':theta,'iter':i,'grad':gradEst,'u':u_m,'sigma2_LL':sigma2LL,'eps':eps}
    subsampleArgs['lambda'] = lambda_use[0]
    # remove some component of the subsample dictionary so that the output is not too large
    subsampleArgs['dev1_sumComponent'] = 0 
    subsampleArgs['dev2_sumComponent'] = 0
    return{'iter':i,'par':finalpar, 'thetaRef':thetaRef, 'acc_rate': acc_rate,'acc_prob':acc_prob,'eps':eps_keep,'sigmaHat':sigmaHat_keep ,
    'nsteps':L_keep,'M':M,'runTime':timePerIter,'args':{'adapt':adaptArgs,'hmc':hmcArgs,'sub':subsampleArgs},
    'current':currentSet, 'signL':signL,'u0':u0,'un':u_m}

#-------------------------------------------------------------------------------
def hmc_within_gibbs(y,x, theta, thetaRef, burnin,samples,hmcArgs,priorArgs,subsampleArgs,adaptArgs,logFile,saveTempOutput=False):
    """ An attempt to incorporate subsampling into HMC
        A new subsample is drawn at each iteration, accept/ reject and then use that subsample to update theta
        estimate the gradient using a subsample, and also estimate the Hamiltonian using subsample (with and without bias correction)
        Implement with adaptive eps. Fix trajectory length (L*eps)
    """
    # eps is stepsize
    # L is number of steps
    # M is covariance matrix of p
    # correlatedP is a dictionary: true/false: whether we correlate p or not, rho
    # set up
    n= len(y)
    niter =  burnin + samples
    npar = len(theta)
    #x = data[:,1:]
    #----------------------------------#
    # arguments for prior
    pfamily = priorArgs['family']
    priorPar1 = priorArgs['par1']
    priorPar2 = priorArgs['par2']
    
    #----------------------------------#
    # HMC arguments
    meanp = np.zeros(npar)
    eps = hmcArgs['eps']
    trajLength = hmcArgs['trajLength']
    L= int(round(trajLength[0]/eps,0))
    M = hmcArgs['pCov'] 
    maxSteps = hmcArgs['maxSteps']
    
    #--------------------------------#
    # subsampling arguments
    #m,nblock,biascorrect
    m = subsampleArgs['subsize']
    #nblocks = subsampleArgs['nblocks']
    rho = subsampleArgs['rho']
    nblocks = 1/(1-rho)
    updateFreq = subsampleArgs['updateFreq'] #frequency of updating reference theta
    updateU = subsampleArgs['updateU']
    biascorrect = subsampleArgs['biasCorrect']
    cvorder = subsampleArgs['order']
    #dev1_i = devll(data,thetaRef)
    #dev1_all = np.sum(dev1_i,axis = 0) #if cannot fit all dev1, switch to calculate sum(dev1)
    if(cvorder >0):
        dev1_all = sumDev(y,x,thetaRef)
    else:
        dev1_all =0 
    if (cvorder ==2):
        dev2_all = sumHessian(x, thetaRef)
    else:
        dev2_all= 0
    llRef = loglike_all(y,x,thetaRef)
    
    #------------------------------------------#
    # parameter for adaptive updating of epsilon
    if(adaptArgs['adapt']==True):
        Hbar = 0 #Hbar is a sumstat ~ different between the desired acceptance rate and the mean acceptance rate upto time t
        alpha = adaptArgs['alpha'] #desired acceptance rate
        gamma =adaptArgs['gamma']
        kappa = adaptArgs['kappa']
        t0 = adaptArgs['t0']
        updateM = adaptArgs['updateM']
        phaseStartPt = adaptArgs['phaseStartPt']
        if(len(phaseStartPt) != len(trajLength)):
            print('phaseEndPt must have same length as trajLength')
        mu = np.log(10*eps)
        logEps = np.log(eps)
        logEpsBar = 0
        eps_keep = np.zeros(niter)
        
    else:
        eps_keep = eps
    phase = 0
    currentTrajLength = trajLength[phase]
    #-----------------------------------#
    # allocate memory for outcome
    theta_keep = np.zeros([niter,npar])
    
    sigmaHat_keep = np.zeros([niter,2]) # after each step
    acc_rate = np.zeros([niter,2]) # first column for u, second column for theta
    L_keep = np.zeros(niter)
    distanceRef = np.zeros(niter) 
    HDiff = np.zeros(niter)
    EDiff = np.zeros(niter)
    llEstPropose = np.zeros(niter) 
    acc_prob = np.zeros([niter,2])   
    timePerIter = np.zeros([niter,2])
    timeUpdateM = 0
    #m_use = np.zeros(niter)
    #---------------------------------------------#
    # initialize
    
    u_m, group_indicator = init_u(m,n,'Approx',rho = rho)
    #if(len(ustart)>0):
    #    u_m = copy.copy(ustart)
    u0 = copy.deepcopy(u_m)
    y_sub = y[u_m]
    x_sub = x[u_m]
    #dev1_sub = dev1_i[u_m]
    if(cvorder ==0):
        dev1_sub = 0
    else:
        dev1_sub = devll(y_sub,x_sub,thetaRef)
    dev2_sub =0
    if(cvorder ==2):
        dev2_sub = np.array(map(hessianll,x_sub,thetaRef*np.ones([m,npar])))
    
    diff = diff_ind(y_sub,x_sub,theta,thetaRef,dev1_sub,dev2_sub,order =cvorder)
    sumGrad = sumGradU_est(theta,thetaRef,dev1_all,dev2_all,order = cvorder) #sum of grad at reference value
    sigma = (n**2/m*np.var(diff))*biascorrect
    
    # U = -(loglike_Est-0.5*sigma_LL + logPrior)
    potentialEst = -loglike_est(theta,thetaRef,llRef,dev1_all,dev2_all,diff,n,cvorder) - dprior(theta,pfamily,priorPar1,priorPar2)
    dev2current=0
    print("Start HMCECS")
    try:
        for i in range(0,niter):
            numerror = False
            progress = i*100/niter
            #if np.mod(progress,5)==0:
            #    print(str(progress)+ "% ",end= "")
            if (np.mod(progress,10)==0 and i>0):
                msg = str(progress) + "% ; nsteps now is: " + str(L) + "; mean acc_u is: " + str(np.mean(acc_prob[:i,0])) + ' and acc_theta: ' + str(np.mean(acc_prob[:i,1]))
                print(msg)
                lf = open(logFile,"a")
                lf.write(msg+ '\n')
                lf.close()
                if(saveTempOutput):
                    part = int(progress*0.1)
                    temp = {'par':theta_keep[:i],'eps':eps_keep,'M':M}
                    np.save('output/temp'+ str(part) + '.npy',temp)
            
            #---------------------------------------------------------------------------------    
            if(np.mod(i,updateFreq)==0 and i>=updateFreq and i<=burnin and (adaptArgs['updateRef']==True)):
                startT = time.time()
                # reset reference
                if (i == updateFreq):
                    thetaRef = np.mean(theta_keep[int(0.5*i):i,:],axis =0)
                else:
                    thetaRef = np.mean(theta_keep[int(0.7*i):i,:],axis =0)
                
                #dev1_i = devll(data,thetaRef)
                #dev1_all = np.sum(dev1_i,axis = 0)
                if(cvorder > 0):
                    dev1_all = sumDev(y,x,thetaRef)
                    dev1_sub = devll(y_sub,x_sub,thetaRef)
                
                dev2_all = sumHessian(x, thetaRef)
                
                if(cvorder ==2):
                    dev2_sub = np.array(map(hessianll,x_sub,thetaRef*np.ones([m,npar]))) #equivalent to hesianll2
                llRef = loglike_all(y,x,thetaRef)
                diff = diff_ind(y_sub,x_sub,theta,thetaRef,dev1_sub,dev2_sub,cvorder)
                sumGrad = sumGradU_est(theta,thetaRef,dev1_all,dev2_all,cvorder)
                sigma = (n**2/m*np.var(diff))*biascorrect
                
                potentialEst = -loglike_est(theta,thetaRef,llRef,dev1_all,dev2_all,diff,n,cvorder) - dprior(theta,pfamily,priorPar1,priorPar2)
                #-------------------------#
                # update M
                if(adaptArgs['updateM']):
                    
                        
                        if(adaptArgs['diagM']==True):
                            var_theta = np.var(theta_keep[int(i/2):i,:],axis = 0)
                            M= np.diag(1/var_theta)
                        else:
                            M = -(dev2_all) - hessianPrior(thetaRef,pfamily,priorPar1,priorPar2)
                            if(np.all(np.linalg.eigvals(M) > 0) == False):
                                var_theta = np.var(theta_keep[int(i/2):i,:],axis = 0)
                                M= np.diag(1/var_theta)
                stopT = time.time()
                timeUpdateM = timeUpdateM +  stopT -startT     
                
            #------------------------------#
            # step1 : update u
            if(updateU):
                startT = time.time()
                potentialCurrent = potentialEst
                diffcurrent = diff
                sigmacurrent = sigma
                ucurrent = u_m
                
                if(nblocks>1):
                    u_m,indexToUpdate = uProp_given_uCurr(m, n, ucurrent,'Approx',lambda_ = nblocks, groupindicators= group_indicator)
                    blockSize = np.shape(indexToUpdate)[1]
                else:
                    # update all u
                    u_m = np.random.choice(n,m,True)
                    indexToUpdate = range(m)
                    blockSize=m
                #accept/reject
                if(cvorder >0):
                    dev1current = copy.copy(dev1_sub[indexToUpdate])
                if(cvorder==2):
                    dev2current = copy.copy(dev2_sub[indexToUpdate])
                
                y_sub = y[u_m]
                x_sub = x[u_m]
                if(cvorder >0):
                    dev1_sub[indexToUpdate] = devll(y_sub[indexToUpdate],x_sub[indexToUpdate],thetaRef)
                if(cvorder==2):
                    dev2_sub[indexToUpdate] = np.array(map(hessianll,x_sub[indexToUpdate],thetaRef*np.ones([blockSize,npar])))
                
                diff = diff_ind(y_sub,x_sub,theta,thetaRef,dev1_sub,dev2_sub,cvorder)
                sigma = (n**2/m*np.var(diff))*biascorrect
                
                potentialEst = -loglike_est(theta,thetaRef,llRef,dev1_all,dev2_all,diff,n,cvorder) - dprior(theta,pfamily,priorPar1,priorPar2)
                # accept new u with probability min(1,posterior_propose/posterior_current), since momentum remained
                # where log(posterior_propose/posterior_current) = -log(post_current) - (-log(post_propose)) = U_current - U_propose
                
                accrate_u = np.exp(np.min([0,-(potentialEst+ 0.5*sigma)+(potentialCurrent+ 0.5*sigmacurrent)]) )
                reject_u = (np.random.uniform(0,1,1)>accrate_u)
                acc_rate[i,0] = 1-reject_u
                acc_prob[i,0] = accrate_u
                if reject_u:
                    potentialEst = potentialCurrent
                    u_m = ucurrent
                    diff = diffcurrent
                    sigma = sigmacurrent
                    if(cvorder >0):
                        dev1_sub[indexToUpdate] = dev1current
                    if(cvorder==2):
                        dev2_sub[indexToUpdate] = dev2current
                
                y_sub = y[u_m]
                x_sub = x[u_m]
                sigmaHat_keep[i,0] = n**2/m*np.var(diff) #sigma
                stopT = time.time()
                timePerIter[i,0] = stopT -startT
            #if i==0:
            #    ustart = u_m
            #------------------------------#
            # step2 : update theta|u
            # skip the correlated p part, not necessary for now
            # sample new momentum
            
            # set current value
            startT = time.time()
            p = np.random.multivariate_normal(meanp,M,1)[0]
            
            thetacurrent = theta
            potentialCurrent = potentialEst
            diffcurrent = diff
            sigmacurrent = sigma
            Ecurrent = potentialCurrent + kinetic(p,M) #energy current
            Hcurrent = Ecurrent + 0.5*sigmacurrent
            L_keep[i] = L
            
            sumGradCurrent = sumGrad
            
            # fist move p half step
            if(biascorrect == True):
                gradEst,s2_temp,dtemp = gradU_estWithCorrection(y_sub,x_sub,thetacurrent,thetaRef,dev1_sub,dev2_sub,sumGradCurrent,n,pfamily,priorPar1,priorPar2,cvorder)
            else:
                gradEst = gradU_est(y_sub,x_sub,theta,thetaRef,dev1_sub,dev2_sub,sumGrad,n,pfamily,priorPar1,priorPar2,cvorder)
            
            p = p-0.5*eps*gradEst  
            
            for s in range(0,L):
                    # move position
                theta = theta+ eps*np.linalg.solve(M,p)
                sumGrad = sumGradU_est(theta,thetaRef,dev1_all,dev2_all,cvorder)
                    
                #move momentum
                if(biascorrect ==True):
                    gradEst,s2_temp,dtemp = gradU_estWithCorrection(y_sub,x_sub,theta,thetaRef,dev1_sub,dev2_sub,sumGrad,n,pfamily,priorPar1,priorPar2,cvorder)
                else:
                    gradEst = gradU_est(y_sub,x_sub,theta,thetaRef,dev1_sub,dev2_sub,sumGrad,n,pfamily,priorPar1,priorPar2,cvorder)
                
                if((np.any(np.isnan(gradEst))==True)or s2_temp >1000 ):
                    accrate = 0
                    reject = True
                    numerror = True
                    break    
                if s < (L-1):
                    p = p- eps*gradEst
                else:
                    p = p-0.5*eps*gradEst
                
            # negate p (not necessary but keep in case change kinetic energy)
            if(numerror == False):
                p  = -p
                
                diff = dtemp#diff_ind(y_sub,x_sub,theta,thetaRef,dev1_sub,dev2_sub,cvorder)
                sigma = (n**2/m*np.var(diff))*biascorrect
                llEstPropose[i] = loglike_est(theta,thetaRef,llRef,dev1_all,dev2_all,diff,n,cvorder)
                potentialEst = -llEstPropose[i]  - dprior(theta,pfamily,priorPar1,priorPar2)
                Epropose = potentialEst + kinetic(p,M)
                
                H =Epropose + 0.5*sigma # H = E + 0.5*sigma^2LL
                
                
                EDiff[i] = Ecurrent - Epropose 
                HDiff[i] = Hcurrent - H
                
                
                accrate = np.exp(np.min([0,HDiff[i]]))
                reject = np.random.uniform(0,1,1)>accrate
            acc_rate[i,1] = 1-reject
            acc_prob[i,1] = accrate
            if reject:
                theta = thetacurrent
                potentialEst = potentialCurrent
                diff = diffcurrent
                sigma = sigmacurrent
                sumGrad = sumGradCurrent
            
            theta_keep[i]=theta                
            distanceRef[i] = np.linalg.norm(theta-thetaRef)
            sigmaHat_keep[i,1] = sigma
            stopT = time.time()
            timePerIter[i,1] = stopT- startT    
            
            #-----------------------------------------#
            # updating eps and L on the fly
            
            if(burnin >0):
                t = i+ 1
                if(adaptArgs['adapt']==True):
                    if(t<= burnin):
                    
                        Hbar = (1-1/(t+t0))*Hbar + 1/(t+t0)*(alpha-accrate)
                        logEps = mu - np.sqrt(t)/gamma*Hbar
                        logEpsBar = t**(-kappa)*logEps + (1-t**(-kappa))*logEpsBar
                        eps = np.min([np.exp(logEps),adaptArgs['maxEps'],currentTrajLength])
                        if (eps <0.00001):
                            print('epsilon is getting small!' + str(eps))
                            # if eps too small replace with original setting
                            #eps = hmcArgs['eps']   
                        if(eps <1e-8):
                            print('epsilon too small!')
                            break                   
                        eps_keep[t] = eps
                        
                        if(phase < len(phaseStartPt)):
                            if((i+1)==phaseStartPt[phase]):
                                currentTrajLength = trajLength[phase]
                                phase +=1 #next phase is phase 1
                        L = min(maxSteps,int(round(currentTrajLength/eps,0)))
                        if(t==burnin) :
                            # t == burnin
                            eps = np.min([np.exp(logEpsBar),adaptArgs['maxEps'],currentTrajLength])
                            eps_keep[t:] = eps
                            L = min(maxSteps,int(round(trajLength[-1]/eps,0)))
                            L_fix = max(1,L)
                        if L==0:
                            print('number of steps reached 0 !')
                            L+=1
                        
                    else:
                        L = L_fix
                else:
                    if t < burnin :
                        L = min(maxSteps,int(round(trajLength[0]/eps,0)))
                    else:
                        L = min(maxSteps,int(round(trajLength[1]/eps,0)))
    except Warning,w:
        print (str(w))
    except TypeError,e:
        print('type error' + str(e))
    except ValueError,e:
        print('value error'+ str(e))
    except IndexError,e:
        print('Index error'+ str(e))
    except (KeyboardInterrupt, SystemExit):
        print('Bye')
    lf = open(logFile,"a")
    lf.write('Run completed successfully')
    lf.close()
    
    finalpar = {'theta':theta_keep}
    currentSet = {'theta':theta,'iter':i,'grad':gradEst,'u':u_m,'sigma2_LL':sigma}
    
    return{'par':finalpar, 'thetaRef':thetaRef, 'acc_rate': acc_rate,'acc_prob':acc_prob,'eps':eps_keep,'sigmaHat':sigmaHat_keep ,'Ediff' : EDiff,
    'nsteps':L_keep,'M':M,'distRef':distanceRef,'Hdist':HDiff,'llEstPropose':llEstPropose,'runTime':timePerIter,'args':{'sub':subsampleArgs,'hmc':hmcArgs},
    'current':currentSet,'timeUpdateM':timeUpdateM,'u0':u0,'un':u_m}

#------------------------------------------------------------#
def gradU_estWithCorrection(y_u,x_u,theta,thetaRef,dev1_u,dev2_u,sumGrad,n,prior,priorPar1,priorPar2,order =2):
    """ FUNCTION TO ESTIMATE GRADIENT USING PROXY , WITH BIAS CORRECTION
        
    """
    # dev1_theta: gradient evaluate at current theta for i in u_m
    m = len(y_u)
    #npar = len(theta)
    npar = np.shape(x_u)[1]
    dev1_theta = devll(y_u,x_u,theta)
    dev1_sum = np.sum((dev1_theta-dev1_u),axis =0)
    dev2_sum = np.sum(dev2_u,axis=0)
    diff_u = diff_ind(y_u,x_u,theta,thetaRef,dev1_u,dev2_u,order)
    sigma = n**2/m*np.var(diff_u)
    # bias correction is 1/2sigma_LL so we add the derivative of the bias correction
    # grad(d_k) = grad(l_k(theta)) - grad(l_k(theta_bar)) - H_i(theta_bar)(theta-theta_bar)
    grad_d_k =  dev1_theta-dev1_u - (theta-thetaRef).dot(dev2_u)
    tmp1 = diff_u-np.mean(diff_u)
    tmp2 = grad_d_k- np.mean(grad_d_k,axis = 0)
    if(order ==2):
        gradll  = -(sumGrad + n/m*(dev1_sum - dev2_sum.dot(theta-thetaRef))) + n**2/(m**2)*np.sum(np.array(map(np.multiply,tmp1,tmp2)),axis = 0)
    else:
        # order = 1 or 0
        gradll  = -(sumGrad + n/m*(dev1_sum )) + n**2/(m**2)*np.sum(np.array(map(np.multiply,tmp1,tmp2)),axis = 0)
    
    gPrior = gradPrior(theta,prior,npar,priorPar1,priorPar2)
    out = gradll  - gPrior
    return out,sigma,diff_u   

       
#--------------------------------------------------------------------------------------#
def gradU_est(y_sub,x_sub,theta,thetaRef,dev1_u,dev2_u,sumGrad,n,prior,priorPar1,priorPar2,order = 2):
    """ FUNCTION TO ESTIMATE GRADIENT USING PROXY, without bias correction """
    # dev1_theta: gradient evaluate at current theta for i in u_m
    m = len(y_sub)
    #npar = len(theta)
    
    dev1_theta = devll(y_sub,x_sub,theta)
    dev1_sum = np.sum((dev1_theta-dev1_u),axis =0)
    dev2_sum = np.sum(dev2_u,axis=0)
    if(order ==2):
        gradll  = -(sumGrad + n/m*(dev1_sum - dev2_sum.dot(theta-thetaRef)))
    else:
        # order = 1 or 0
        gradll  = -(sumGrad + n/m*(dev1_sum ))
    gPrior = gradPrior(theta,prior,len(theta),priorPar1,priorPar2)
    out = gradll  - gPrior
    return out  
              
                            
#------------------------------------------------------------------#              
                            
def getPrior(priorFam,npar,beta = 0,M = 0,p1 = 0,p2 = None,nu = 20,scale = 10):
    if(priorFam == 'gaussian'):
        if(p1==0):
            p1 = np.zeros(npar) #priorMean
            # else p1 is input
        if(p2 is None):
            p2 = scale**2*np.identity(npar) #priorCov
            # else p2 is input
    else:
        print('invalid options')
        p1 =p2 =  np.nan
    
    priorInfo = {'family':priorFam, 'par1':p1,'par2':p2,'beta':beta} 
    return priorInfo
    
#---------------------------------------------------------------#
def simData(npar,n):
    dataFolder = 'data'+str(npar)+ 'Par/'
    dataFile = 'data'+str(npar)+ 'Par'
    if not os.path.exists(dataFolder):
        os.makedirs(dataFolder)
    y_i = np.ones(n)
    while sum(y_i) > 0.9*n:    
        beta_true = np.array([random.uniform(-5,5)for i in range(0,npar)])
        
        x_ip = np.ones([n,npar])
        for c in range(1,npar):
            
            x_ip[...,c] = np.random.normal(0,1, n)
        p1_i = 1/(1+np.exp(-np.dot(x_ip,beta_true)))
        y_i = np.random.binomial(1,p1_i)
    
    data_ip = np.column_stack((y_i,x_ip))
    
    np.savetxt(dataFolder + dataFile + '.txt',data_ip)
    np.savetxt(dataFolder + 'betaTrue' + '.txt',beta_true)
    
    

def momentEstimator(theta,sign,order=1):
    # theta and sign are vector of same length
    # default order =1 (expectation)
    sumSign = np.sum(sign)
    moments_approx = np.sum((theta**order)*sign)/sumSign
    return moments_approx
    


#-----------------------------------------------------------------------------#
# Implementation of SG-HMC
def gradEstSG(y,x,theta,n):
    """ 
    function that estimates gradient by subsampling with no control variate (i.e n/m*sum(grad))
    """
    gradEst = n/len(y)*np.sum(devll(y,x,theta),axis = 0)
    return gradEst

#------------------------------------------------------------------------#
def sghmcMat(y,x, theta,thetaRef,burnin,samples,eps,trajLength,M,sgpar,priorArgs,adaptArgs,CV=2,logFile='test.txt',saveTempOutput=False):
    """ IMPLEMENTATION OF SGHMC with matrix SGHMC parameters
        sgpar is a dictionary of parameters for SG-HMC including eta,alpha and beta
    """
    
    niter = burnin + samples
    npar = len(theta)
    
    n = len(y)
    theta_keep = np.zeros([niter,npar])
    timePerIter = np.zeros(niter)
    timeUpdatePar = 0
    # arguments for prior
    pfamily = priorArgs['family']
    priorPar1 = priorArgs['par1']
    priorPar2 = priorArgs['par2']
    
    meanp = np.zeros(len(theta))
    m = sgpar['subsize']
    C = sgpar['C']
    Bhat = 0.5*eps* sgpar['V']
    W = C- Bhat
    if (np.min(np.linalg.eigvals(W))<0):
        print('Warnings: C needs to be > Bhat')
    #-------------------------------#
    if(CV >0 ):
        dev1_all = sumDev(y,x,thetaRef)
    if(CV==2):
        dev2_all = sumHessian(x, thetaRef)
    #llRef = loglike_all(y,x,thetaRef)
    u_m = np.random.choice(n,m,True)
    y_sub = y[u_m]
    x_sub = x[u_m]
    if(CV >0):
        dev1_sub = devll(y_sub,x_sub,thetaRef)
    else:
        dev1_sub = 0
    if(CV==2):
        dev2_sub = np.array(map(hessianll,x_sub,thetaRef*np.ones([m,npar])))
    phase = 0
    L = int(round(trajLength[phase]/eps,0))
    print('start SGHMC')
    try:
        for i in range(0,niter):
            progress = i*100/niter
            if np.mod(progress,5)==0:
                print(str(progress)+ "% ",end= "")
            if (np.mod(progress,10)==0 and i>0):
                msg = str(progress) + "%"
                lf = open(logFile,"a")
                lf.write(msg+ '\n')
                lf.close()
                if(saveTempOutput):
                    part = int(progress*0.1)
                    temp = {'par':theta_keep[:i],'M':M,'eps':eps,'L': L}
                    np.save('output/temp'+ str(part) + '.npy',temp)
            
        
            if(np.mod(i,adaptArgs['updateFreq'])==0 and i>=adaptArgs['updateFreq'] and i<=burnin):
                startT = time.time()
                # reset reference
                if (i == adaptArgs['updateFreq']):
                    thetaRef = np.mean(theta_keep[int(0.7*i):i,:],axis =0)
                else:
                    thetaRef = np.mean(theta_keep[int(0.9*i):i,:],axis =0)
                
                if(CV>0):
                    dev1_all = sumDev(y,x,thetaRef)
                    dev1_sub = devll(y_sub,x_sub,thetaRef)
                dev2_all = sumHessian(x, thetaRef)
                
                if(CV==2):
                    dev2_sub = np.array(map(hessianll,x_sub,thetaRef*np.ones([m,npar])))
                  
                
              #  update M          
                if(adaptArgs['updateM']):
                    
                    M = -(dev2_all) - hessianPrior(thetaRef,pfamily,priorPar1,priorPar2)
                    if(adaptArgs['diagM']==True):
                        M = np.diag(np.diag(M))
                    if(np.all(np.linalg.eigvals(M) > 0) == False):
                        var_theta = np.var(theta_keep[int(i/2):i,:],axis = 0)
                        M= np.diag(1/var_theta) 
                                  
                stopT = time.time()
                timeUpdatePar = timeUpdatePar +  stopT -startT                         
            
            startT = time.time()   
            
            
            p = np.random.multivariate_normal(meanp,M,1)[0]
            for s in range(0,L):
                u_m = np.random.choice(n,m,True)    
                y_sub = y[u_m]
                x_sub = x[u_m] 
                  
                #update theta
                theta = theta + eps*np.linalg.solve(M,p)
                
                #update p
                if (CV==2):       
                           
                    dev1_sub = devll(y_sub,x_sub,thetaRef)
                    dev2_sub = np.array(map(hessianll,x_sub,thetaRef*np.ones([m,npar])))
            # move momentum
                    sumGrad = sumGradU_est(theta,thetaRef,dev1_all,dev2_all)
                    gradUEst = gradU_est(y_sub,x_sub,theta,thetaRef,dev1_sub,dev2_sub,sumGrad,n,pfamily,priorPar1,priorPar2)
                    
                else:
                    if(CV==1):
                        # use the control variate in the Baker paper
                        gradUEst = -dev1_all - n/m*np.sum(devll(y_sub,x_sub,theta)-devll(y_sub,x_sub,thetaRef),axis = 0) - gradPrior(theta,pfamily,npar,priorPar1,priorPar2) 
                    else:
                        gradUEst = - n/m*np.sum(devll(y_sub,x_sub,theta),axis = 0)  - gradPrior(theta,pfamily,npar,priorPar1,priorPar2) 
                
                p = p - eps*gradUEst - eps*np.dot(C,np.linalg.solve(M,p)) + np.random.multivariate_normal(meanp,2*W*eps,1)[0]
                
                # move position
                               
            stopT = time.time()
            timePerIter[i] = stopT- startT    
            
            theta_keep[i]= theta
            
            if((i <burnin) and (adaptArgs['adapt']==True) ):
                if ((phase < len(adaptArgs['phaseStartPt'])) and  ((i+1)==adaptArgs['phaseStartPt'][phase])):
                    L = int(trajLength[phase]/eps)
                    phase = phase+1
            
    except Warning,w:
        print (str(w))
    except TypeError,e:
        print('type error' + str(e))
    except ValueError,e:
        print('value error'+ str(e))
    except IndexError,e:
        print('Index error'+ str(e))
    except (KeyboardInterrupt, SystemExit):
        print('Bye')
    lf = open(logFile,"a")
    lf.write('Run completed successfully')
    lf.close()

    finalpar = {'theta':theta_keep}
    currentSet = {'theta':theta,'iter':i,'grad':gradUEst}
    return {'par':finalpar,'eps':eps,'nsteps':L,'args':sgpar,'runTime':timePerIter,'current':currentSet,'M':M,'timeUpdatePar':timeUpdatePar}
 
  
#--------------------------------------------------------------#
# SGLD
def sgld(y,x, theta,thetaRef,burnin,samples,eps,subsize,priorArgs,adaptArgs,CV,logFile,saveTempOutput=False):
    """ IMPLEMENTATION OF SGHMC with matrix SGHMC parameters
        sgpar is a dictionary of parameters for SG-HMC including eta,alpha and beta
    """
    
    niter = burnin + samples
    npar = len(theta)
    
    n = len(y)
    theta_keep = np.zeros([niter,npar])
    timePerIter = np.zeros(niter)
    # arguments for prior
    pfamily = priorArgs['family']
    priorPar1 = priorArgs['par1']
    priorPar2 = priorArgs['par2']
    #
    m = subsize
    
    #-------------------------------#
    dev1_all = sumDev(y,x,thetaRef)
    if (CV==True):
        dev2_all = sumHessian(x, thetaRef)
    #llRef = loglike_all(y,x,thetaRef)
    u_m = np.random.choice(n,m,True)
    y_sub = y[u_m]
    x_sub = x[u_m]
    dev1_sub = devll(y_sub,x_sub,thetaRef)
    dev2_sub = np.array(map(hessianll,x_sub,thetaRef*np.ones([m,npar])))
    try:
        for i in range(0,niter):
            progress = i*100/niter
            if np.mod(progress,5)==0:
                print(str(progress)+ "% ",end= "")
            if (np.mod(progress,10)==0 and i>0):
                msg = str(progress) + "%"
                lf = open(logFile,"a")
                lf.write(msg+ '\n')
                lf.close()
                if(saveTempOutput):
                    part = int(progress*0.1)
                    temp = {'par':theta_keep[:i]}
                    np.save('output/temp'+ str(part) + '.npy',temp)
            
            
            startT = time.time()   
                        
            u_m = np.random.choice(n,m,True)           
            y_sub = y[u_m]
            x_sub = x[u_m]          
            dev1_sub = devll(y_sub,x_sub,thetaRef)
            if(CV==True):
                dev2_sub = np.array(map(hessianll,x_sub,thetaRef*np.ones([m,npar])))
                # move momentum
                sumGrad = sumGradU_est(theta,thetaRef,dev1_all,dev2_all)
                
                gradUEst = gradU_est(y_sub,x_sub,theta,thetaRef,dev1_sub,dev2_sub,sumGrad,n,pfamily,priorPar1,priorPar2)
                
            else:
                # first order
                gradUEst = -dev1_all - n/m*np.sum(devll(y_sub,x_sub,theta)-dev1_sub,axis = 0) - gradPrior(theta,pfamily,npar,priorPar1,priorPar2) #+ gradPrior(thetaRef,pfamily,npar,priorPar1,priorPar2)
                
            theta = theta- 0.5*eps*gradUEst+ np.random.multivariate_normal(np.zeros(npar),eps*np.identity(npar),1)[0]
                               
            stopT = time.time()
            timePerIter[i] = stopT- startT    
            
            theta_keep[i]= theta
            
    except Warning,w:
        print (str(w))
    except TypeError,e:
        print('type error' + str(e))
    except ValueError,e:
        print('value error'+ str(e))
    except IndexError,e:
        print('Index error'+ str(e))
    except (KeyboardInterrupt, SystemExit):
        print('Bye')
    lf = open(logFile,"a")
    lf.write('Run completed successfully')
    lf.close()

    finalpar = {'theta':theta_keep}
    currentSet = {'theta':theta,'iter':i,'grad':gradUEst}
    return {'par':finalpar,'eps':eps,'runTime':timePerIter,'current':currentSet}
 
 
#-------------------------------------------------------------------------------#

# maximum likelihood estimator
def minusloglike(theta,*args):
    y = args[0]
    x = args[1]
    out = -loglike_all(y,x,theta)
    return out
def minusDerLL(theta,*args):
    y = args[0]
    x = args[1]
    #npar = len(theta)
    const = 1/(1+np.exp(-np.dot(x,theta)))
    dev_ind = y.reshape(-1,1)*x - const.reshape(-1,1)*x
    gradll= -np.sum(dev_ind,axis =0) 
    return gradll
def minusHess(theta,*args):
    y = args[0]
    x = args[1]
    #npar = len(theta)
    sumH = sumHessian(x, theta)
    h2 = -sumH
    return h2
    
#------------------------------------------#

# maximum posterior estimator- with a N(0,scale^2) prior
def minuslogpost(theta,*args):
    y = args[0]
    x = args[1]
    scale = args[2]
    npar = len(theta)
    out = -loglike_all(y,x,theta) - dprior(theta,'gaussian',np.zeros(npar),scale**2*np.identity(npar))
    return out
def minusDerPost(theta,*args):
    y = args[0]
    x = args[1]
    scale = args[2]
    npar = len(theta)
    const = 1/(1+np.exp(-np.dot(x,theta)))
    dev_ind = y.reshape(-1,1)*x - const.reshape(-1,1)*x
    gradlp= -np.sum(dev_ind,axis =0) - gradPrior(theta,'gaussian',npar,np.zeros(npar),scale**2*np.identity(npar))
    return gradlp
def minusHessPost(theta,*args):
    y = args[0]
    x = args[1]
    scale = args[2]
    npar = len(theta)
    sumH = sumHessian(x, theta)
    h2 = -sumH - hessianPrior(theta,'gaussian',np.zeros(npar),scale**2*np.identity(npar))
    return h2
    
#-------------------------------------------#
def find_opt(gamma, algorithm, rho,intitial=50000):
    # return optimal lambda for exact and optimal m for approximate algorithm
    if(algorithm == 'Exact'):
        minlambda = int(np.round(1/(1-rho)))
        lambda_opt = int(np.ceil(np.exp(-0.1022 + 0.4904*np.log(np.max(gamma)))/minlambda))*minlambda
        return lambda_opt
    elif(algorithm=='Approx'):
        min_m = int(np.round(1/(1-rho)))
        m_opt = int(np.ceil(scipy.optimize.minimize(CT_approx,intitial,args = (14249843,0.99)).x/min_m))*min_m
        return m_opt
def varLogLApprox(gamma,m):
    varl = gamma/m + gamma**2/(2*m**3)
    return varl        
def CT_approx(m,gamma,rho):
    sig_approx = np.sqrt(varLogLApprox(gamma,m))
    IF_approx = IF_factor_exact(rho, sig_approx)
    return m*IF_approx
  
#----------------------------------------------------#
def IF_factor_exact(rho, sig):
    """
    MATLAB code from MNT that I translate to Python
    NOTE: Input here is standard deviation and not variance
    
    NOTE that this is for the case when we have correlation between the estimators. If not, set rho = 0, and we get the expression in Pitt et al.
    """
    #function out = IF_factor(rho,sig)    
    mu_star = -sig*rho*np.sqrt((1-rho)/(1+rho))
    sig2_star = (1-rho)/(1+rho)
    tau = sig*np.sqrt(1-rho**2)
    
    #p = @(w) normcdf(w+tau)-exp(-w*tau-tau^2/2)*normcdf(w);    
    p = lambda w: sps.norm.cdf(w+tau) - np.exp(-w*tau-tau**2/2)*sps.norm.cdf(w)
        
    # p_prime = @(w) normpdf(w+tau)+exp(-w*tau-tau^2/2)*(tau*normcdf(w)-normpdf(w));
    p_prime = lambda w: sps.norm.pdf(w+tau) + np.exp(-w*tau-tau**2/2)*(tau*sps.norm.cdf(w) - sps.norm.pdf(w))
    
    #phi_prime = @(w) -w*normpdf(w);
    phi_prime = lambda w:  -w*sps.norm.pdf(w) 
    
    #p_2prime = @(w) phi_prime(w+tau)-exp(-w*tau-tau^2/2)*(tau^2*normcdf(w)-2*tau*normpdf(w)+phi_prime(w));
    p_2prime = lambda w: phi_prime(w+tau)-np.exp(-w*tau-tau**2/2)*(tau**2*sps.norm.cdf(w)-2*tau*sps.norm.pdf(w)+phi_prime(w))
    
    #p_3prime = @(w) ((w+tau)^2-1)*normpdf(w+tau)+exp(-w*tau-tau^2/2)*(tau^3*normcdf(w)-normpdf(w)*(3*tau^2+3*tau*w+w^2-1));
    p_3prime = lambda w: ((w+tau)**2-1)*sps.norm.pdf(w+tau)+np.exp(-w*tau-tau**2/2)*(tau**3*sps.norm.cdf(w)-sps.norm.pdf(w)*(3*tau**2+3*tau*w+w**2-1))
    
    # p_4prime = @(w) (3*(w+tau)-(w+tau)^3)*normpdf(w+tau)-exp(-w*tau-tau^2/2)*( tau^4*normcdf(w)-normpdf(w)*(4*tau^3+6*tau^2*w+4*tau*w^2-4*tau+w^3-3*w) )
    p_4prime = lambda w:  (3*(w+tau)-(w+tau)**3)*sps.norm.pdf(w+tau)-np.exp(-w*tau-tau**2/2)*(tau**4*sps.norm.cdf(w)-sps.norm.pdf(w)*(4*tau**3+6*tau**2*w+4*tau*w**2-4*tau+w**3-3*w))
    
    #f = @(w) (1+p(w))/(1-p(w));
    f = lambda w:  (1+p(w))/(1-p(w))
   
    #f_prime = @(w) p_prime(w)*(1+f(w))/(1-p(w));
    f_prime = lambda w: p_prime(w)*(1+f(w))/(1-p(w))
    
    #f_2prime = @(w) ( p_2prime(w)*(1+f(w))+2*p_prime(w)*f_prime(w) )/(1-p(w));
    f_2prime = lambda w: (p_2prime(w)*(1+f(w))+2*p_prime(w)*f_prime(w) )/(1-p(w))
    
    #f_3prime = @(w) ( p_3prime(w)*(1+f(w))+3*p_2prime(w)*f_prime(w)+3*p_prime(w)*f_2prime(w) )/(1-p(w));
    f_3prime = lambda w: (p_3prime(w)*(1+f(w))+3*p_2prime(w)*f_prime(w)+3*p_prime(w)*f_2prime(w) )/(1-p(w));
    
    #f_4prime = @(w) ( p_4prime(w)*(1+f(w))+4*p_3prime(w)*f_prime(w)+6*p_2prime(w)*f_2prime(w)+4*p_prime(w)*f_3prime(w) )/(1-p(w));
    f_4prime = lambda w: (p_4prime(w)*(1+f(w))+4*p_3prime(w)*f_prime(w)+6*p_2prime(w)*f_2prime(w)+4*p_prime(w)*f_3prime(w) )/(1-p(w))
    
    ans = f(mu_star)+1/2*f_2prime(mu_star)*sig2_star+1/8*f_4prime(mu_star)*sig2_star**2
    if np.isinf(ans) or np.isnan(ans):
	print ("Inf or Nan in IF expression. Variance is %s" % sig**2)
	print ("Stir optimizer away from here")
	return 10e100 
    else:
	return f(mu_star)+1/2*f_2prime(mu_star)*sig2_star+1/8*f_4prime(mu_star)*sig2_star**2
  

    