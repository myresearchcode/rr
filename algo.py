import numpy as np;
from numpy import sqrt,square,log;
from time import time;
from scipy.linalg import norm;
import logging;

logger = logging.getLogger(__name__)

def sca(Unrated_M,SiM,m_ind,mind_R,K): #greedy and b_greedy are in fact same.
    if (Unrated_M.size <= K):
        K = Unrated_M.size
    sca_func = {
                'risks'  : np.exp,  #seek seeking
            }
    c_f = 'risks'
    logger.info('Risk Seeking Top K Recommendation: %s' %(c_f))
    eps = 1e-8
    Rec_Movies_ID = list() #Greedy Set
    CG = list()
    sum_W = np.zeros((Unrated_M.shape[0],m_ind.shape[0]))
    Tsum_W = sum_W + SiM[Unrated_M[:,None],m_ind]
    if c_f == 'risks':
        xf = -(np.dot(1-np.exp(1*Tsum_W),mind_R) + eps)
        
    F_j = np.copy(xf)  #F_i for all the items in the unrated set
    xf_b = 0
    for i in xrange(K):  #iterate over the set of unrated movies.
        CG.append(1-np.min((xf-xf_b)/F_j))
        ind = np.argmax(xf)
        sum_W = Tsum_W[ind,:]
        best_select = Unrated_M[ind]
        Rec_Movies_ID.append(best_select) #We add the best selected to the array.
        Unrated_M  = np.delete(Unrated_M,ind)   #this is ok, since indexes are unique
        F_j = np.delete(F_j,ind)  #remove the item selected by the greedy strategy
        xf_b = np.delete(xf,ind)
        Tsum_W = sum_W + SiM[Unrated_M[:,None],m_ind]
        if c_f == 'risks':
            xf = -(np.dot(1-np.exp(1*Tsum_W),mind_R) + eps)
    logger.info('Done with risk seeking top-k recommendations')
    return np.array(Rec_Movies_ID,dtype=int),max(CG)    #this gives sorted list
