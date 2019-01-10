import numpy as np;
from scipy import linalg;
from scipy.optimize import nnls;
from numpy import dot;
from os.path import isfile;
#import timeit
import logging

logger = logging.getLogger(__name__)

def nmf_mup(R, folder_path, split, rank = 50, max_iter=50, error_limit=1e-4, fit_error_limit=1e-6):
    """ Decompose R to X*Y using multiplicative update method"""
    uf_file = "mup_user_feature_%d.txt" %(splt);
    mf_file = "mup_movie_feature_%d.txt" %(splt);
    mf_file = 'mup_movie_feature.txt';
    filename2 = path.join(dir_name,mf_file);

    if path.isfile(filename1) and path.isfile(filename2):
        X = np.loadtxt(filename1,dtype=np.float16,delimiter=',');
        Y = np.loadtxt(filename2,dtype=np.float16,delimiter=',');
        return X,Y;
    eps = 1e-5;
    W = np.sign(R); #to use only observed data
    rows, columns = R.shape;
    np.random.seed(1000);
    X = np.random.rand(rows, rank);
    X = np.maximum(X, eps); #take only positive elements

    Y = np.linalg.lstsq(X, R)[0];  #initial Y matrix.  It can be randomized also.  But here it is ||R-XY||_2
    Y = np.maximum(Y, eps); #non-negativity constraints.

    masked_R = W * R;   #R is already masked
    R_est_prev = dot(X, Y); #new R matrix
    for i in xrange(1, max_iter + 1):
        top = dot(masked_R, Y.T);
        bottom = (dot((W * dot(X, Y)), Y.T)) + eps;
        X *= top / bottom;
        X = np.maximum(X, eps);
        top = dot(X.T, masked_R);
        bottom = dot(X.T, W * dot(X, Y)) + eps;
        Y *= top / bottom;
        Y = np.maximum(Y, eps);

        R_est = dot(X, Y);
        err = W * (R_est_prev - R_est);
        fit_residual = np.sqrt(np.sum(err ** 2));
        R_est_prev = R_est;
        curRes = linalg.norm(W * (R - R_est), ord='fro');
        if curRes < error_limit or fit_residual < fit_error_limit:
            break

    #print "Split: %d Rank: %d Total Residual:%f" %(splt, rank, np.round(curRes, 4));
    np.savetxt(filename1,X,fmt='%2.7f',delimiter=',');
    np.savetxt(filename2,Y,fmt='%2.7f',delimiter=',');
    return X,Y

def als_nmf(R, folder_path, split, rank = 50, max_iter=50, error_limit=1e-4, lmbda = 0.01):
    """ Decompose R to UV' using ALS method"""
    uf_file = folder_path + "als_user_feature_%d.txt" %(split);
    mf_file = folder_path + "als_movie_feature_%d.txt" %(split);

    if isfile(uf_file) and isfile(mf_file):
        logging.info("Loading the already generated feature matrices")
        U = np.loadtxt(uf_file,dtype=np.float32,delimiter=',');
        V = np.loadtxt(mf_file,dtype=np.float32,delimiter=',');
        return U,V;
    W = np.sign(R);
    rows, columns = R.shape;
    random = np.random.RandomState(seed=split)
    U = np.zeros((rows,rank));
    V = 5*random.rand(rank,columns);
    V[0,:] = np.mean(R[np.nonzero(R)],axis=0);

    #st = timeit.default_timer()
    for i in xrange(1,max_iter+1):
        for u, Wu in enumerate(W):
            U[u] = nnls(np.dot(V, np.dot(np.diag(Wu), V.T)) + lmbda*np.eye(rank), np.dot(V, np.dot(np.diag(Wu), R[u].T)))[0].T;
            #X[u] = np.linalg.lstsq(np.dot(Y, np.dot(np.diag(Wu), Y.T)) + lmbda*np.eye(rank), np.dot(Y, np.dot(np.diag(Wu), R[u].T)))[0].T;
            #X[u] = np.linalg.solve(np.dot(Y, np.dot(np.diag(Wu), Y.T)) + lmbda*np.eye(rank), np.dot(Y, np.dot(np.diag(Wu), R[u].T))).T;
        for v, Wv in enumerate(W.T):
            V[:,v] = nnls(np.dot(U.T, np.dot(np.diag(Wv), U)) + lmbda* np.eye(rank), np.dot(U.T, np.dot(np.diag(Wv), R[:,v])))[0];
            #Y[:,v] = np.linalg.lstsq(np.dot(X.T, np.dot(np.diag(Wv), X)) + lmbda* np.eye(rank), np.dot(X.T, np.dot(np.diag(Wv), R[:,v])))[0];
            #Y[:,v] = np.linalg.solve(np.dot(X.T, np.dot(np.diag(Wv), X)) + lmbda* np.eye(rank), np.dot(X.T, np.dot(np.diag(Wv), R[:,v])));
        R_est = dot(U,V);
        curRes = linalg.norm(W *(R - R_est), ord = 'fro');
        logging.info("Residue at iteration: %d is %f" %(i,curRes))
        if curRes < error_limit:
            logging.info("Residue less than error limit.... Early exit at iteration: %d " %(i))
            break;

    #et = timeit.default_timer()-st
    #print "Elapsed Time : %f" %(et)
    #print "Split: %d Rank: %d Total Residual: %f " %(splt, rank, np.round(curRes, 4));
    np.savetxt(uf_file,U,fmt='%2.8f',delimiter=',');
    np.savetxt(mf_file,V,fmt='%2.8f',delimiter=',');
    return U,V
