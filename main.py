#!/usr/bin/env python2.7

import sys;
import os.path as path;
import numpy as np;
from time import time;
#import  scipy.spatial.distance as distance;
import logging;
from metrics import get_user_metrics,stratified_recall,n_call_K,catalog_coverage,strat_recall_denm_per_user,serendipity_score
from nmf import als_nmf
from algo import *
from joblib import Parallel,delayed,cpu_count;
from data_utils import *
np.seterr(invalid='raise')
baselines = {
    'sca' : sca,   #Extended Greedy algorithm with collaborative filtering
}

sca_func = 'risks' # can be powe pow1 pow9 sqrt log riska1 riska5 riska01 riskae risks
    
def main(alg): #main function
    max_rat = 5 #maximum utility value
    R = U = V = II_SiM = Mov_Gen_Mat = Rhat = None
    rank = 50  #for matrix factorization.
    tot_itr = 5
    rmd_sz = [3,5,10,20]  #metrics will be calculated for this recommendation.
    K = max(rmd_sz)
    all_datasets = ['dataset'+str(_) for _ in xrange(3,4)]
    for dataset in all_datasets:
        result_dict = dict()
        result_dict['curv'] = list()
        for i in rmd_sz:
            result_dict[i] = dict()
            result_dict[i]['ncall'] = list()
            result_dict[i]['sr'] = list()
            result_dict[i]['cc'] = list()
            result_dict[i]['ild'] = list()
            result_dict[i]['dcg'] = list()
            result_dict[i]['gc'] = list()
            result_dict[i]['feature_dist'] = list()
            result_dict[i]['serendipity_score'] = list()
            result_dict[i]['apk'] = list()
            result_dict[i]['ndcg'] = list()

        folder_path = get_folder_path(dataset)
        R,U_map,M_map = get_rating_matrix(dataset) #U/M_map contains original user/movie id.
        genres = get_genres(dataset)    #we call it topics instead of genres
        curr = time()
        (r_sz,c_sz) = R.shape
        II_SiM = np.zeros((c_sz,c_sz))    #item-item similarity matrix. the pariwise cosine similarity between the V feature vectors
        M = get_movies(dataset,M_map)   #Load the movies file in a structured array.

        Mov_Gen_Mat = build_genre_mat(M,genres);   #Genre Matrix for the movies. It is a binary matrix of M*18 size. we call it topic matrix
        
        for splt in xrange(1,tot_itr+1):    #we try 5 random splits and then average over it
            if alg == 'sca':
                log_file = folder_path + '%s_%s_access_%d.log' %(alg,sca_func,splt)
            else:
                log_file = folder_path + '%s_access_%d.log' %(alg,splt)
                
            logging.basicConfig(filename = log_file, format='%(asctime)s : %(message)s', level=logging.INFO)
            U_rmd = dict()
            U_curv = dict()
            user_metric_dict = dict()
            U_strat_denm = dict()
            for i in rmd_sz:
                user_metric_dict[i] = dict()
                user_metric_dict[i]['relev_set'] = set()
                user_metric_dict[i]['relev_list'] = list()
                user_metric_dict[i]['srn'] = dict()
                user_metric_dict[i]['gc'] = dict()
                user_metric_dict[i]['ild'] = dict()
                user_metric_dict[i]['dcg'] = dict()
                user_metric_dict[i]['feature_dist'] = dict()
                user_metric_dict[i]['serendipity_score'] = dict()
                user_metric_dict[i]['apk'] = dict()
                user_metric_dict[i]['ndcg'] = dict()

            R_train,R_test = split_train_test(R,5.0,splt) #5% is test set.
            users = get_test_users(R_test)  #get only users who has test ratings.
            U,V = als_nmf(R_train,folder_path,splt)
            II_SiM = get_item_item_sim(V.T)  #V is of dimension (k,n)
            if alg == 'mmr' or alg == 'dum' or alg == 'msd' or alg == 'mf':
                Rhat = get_predicted_ratings(U,V,max_rat)

            for u_ind in users:
                m_ind,m_rat = get_rated_movies(R_train,u_ind)   #get already rated movie indxs and ratings values;
                Unrated_M = np.setdiff1d(np.arange(c_sz,dtype=np.uint16),m_ind,assume_unique=True)
                if alg == 'sca' or alg == 'modular':
                    U_rmd[u_ind],U_curv[u_ind] = baselines[alg](Unrated_M,II_SiM,m_ind,m_rat,K);	#Call the function
                else:
                    U_rmd[u_ind],U_curv[u_ind] = baselines[alg](Unrated_M,II_SiM,m_ind,m_rat,K,u_ind,Rhat,Mov_Gen_Mat)
                    
                U_strat_denm[u_ind] = strat_recall_denm_per_user(R_test,u_ind);

                for i in rmd_sz:
                    get_user_metrics(user_metric_dict,U_rmd,U_curv,u_ind,i,R_train,R_test,Mov_Gen_Mat,M,M_map,m_ind,V.T);
            result_dict['curv'].append(np.mean(U_curv.values()))
            logging.info("============================================================================")
            for i in rmd_sz:
                logging.info("Top %d Recommendations" %(i))
                result_dict[i]['ncall'].append(n_call_K(user_metric_dict[i]['relev_list'],len(users)))
                result_dict[i]['sr'].append(stratified_recall(sum(user_metric_dict[i]['srn'].values()), sum(U_strat_denm.values())))
                logging.info("Stratified Recall : %f" %(result_dict[i]['sr'][-1]))
                result_dict[i]['cc'].append(catalog_coverage(len(user_metric_dict[i]['relev_set']),c_sz))
                logging.info("Catalog Coverage : %f" %(result_dict[i]['cc'][-1]))
                result_dict[i]['gc'].append(np.mean(user_metric_dict[i]['gc'].values()))
                logging.info("Genre Coverage : %f" %(result_dict[i]['gc'][-1]))
                result_dict[i]['ild'].append(np.mean(user_metric_dict[i]['ild'].values()))
                logging.info("Average ILD : %f" %(result_dict[i]['ild'][-1]))
                result_dict[i]['dcg'].append(np.mean(user_metric_dict[i]['dcg'].values()))
                logging.info("Average DCG : %f" %(result_dict[i]['dcg'][-1]))
                result_dict[i]['serendipity_score'].append(np.mean(user_metric_dict[i]['serendipity_score'].values()))
                logging.info("Average Serendipity Score : %f" %(result_dict[i]['serendipity_score'][-1]))
                result_dict[i]['feature_dist'].append(np.mean(user_metric_dict[i]['feature_dist'].values()))
                logging.info("Average Feature Distance : %f" %(result_dict[i]['feature_dist'][-1]))
                result_dict[i]['apk'].append(np.mean(user_metric_dict[i]['apk'].values()))
                logging.info("AP@K : %f" %(result_dict[i]['apk'][-1]))
                result_dict[i]['ndcg'].append(np.mean(user_metric_dict[i]['ndcg'].values()))
                logging.info("NDCG : %f" %(result_dict[i]['ndcg'][-1]))
            logging.info("Average Greedy Curvature: %f" %(result_dict['curv'][-1]))    
            logging.info("============================================================================")
            logger = logging.getLogger()
            for hdlr in logger.handlers[:]:
                hdlr.close()
                logger.removeHandler(hdlr)                              
        for i in rmd_sz:
            print "============================================================================";
            print "============================================================================";
            print "%d recommendations for %s algorithm (%d iterations)" %(i,alg,tot_itr);
            print "Averag True Topic Coverage: %f" %(np.mean(result_dict[i]['gc']));
            print "Averag Catalog Coverage: %f" %(np.mean(result_dict[i]['cc']));
            print "Averag ILD: %f" %(np.mean(result_dict[i]['ild']));
            print "Averag Strat Recall: %f" %(np.mean(result_dict[i]['sr']));
            print "Averag n-call@K: %f" %(np.mean(result_dict[i]['ncall']));
            print "Averag DCG: %f" %(np.mean(result_dict[i]['dcg']));
            print "Averag Serendipity Score: %f" %(np.mean(result_dict[i]['serendipity_score']));
            print "Averag Feature Distance: %f" %(np.mean(result_dict[i]['feature_dist']));
            print "MAP: %f" %(np.mean(result_dict[i]['apk']));
            print "NDCG: %f" %(np.mean(result_dict[i]['ndcg']));
    
        print "Average Greedy Curvature: %f" %(np.mean(result_dict['curv']))
        print "============================================================================";
if __name__ == '__main__':
    #Parallel(n_jobs=cpu_count())(delayed(main)(alg) for alg in baselines.keys())
    for alg in baselines.keys():
        main(alg)
