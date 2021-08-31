# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 14:37:29 2021

@author: Agnij
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

anomaly_df = pd.read_csv('anomaly_detection.csv')
anomaly_df['date'] = pd.date_range(start='1/1/2016', periods=len(anomaly_df), freq='D')
anomaly_df.set_index('date', inplace=True)

'''
Assumption - The file contains ordered values in ascending order of dates
             between 01/01/2016 - 30/12/2016
*As mentioned in problem statement
'''

plt.rc('figure',figsize=(12,6))
plt.rc('font',size=15)
anomaly_df.plot()

'''
Reviewed By Analyst for 1st 9 months
found anomalous behaviour between 14th Feb to 21st Feb

for c in list(anomaly_df.columns):
    sns.distplot(anomaly_df[c])

# Sensor EDDAB - shows a gaussian distribution
# Sensor CEACC - shows a positively skewed distribution ( Mean > Median > Mode )
# Sensor FAXAE - shows a negatively skewed distribution ( Mode > Median > Mean )
# Sensor FBFFD - normal
# Sensor CCDEF - right skew
    
sns.relplot(x='CEACC', y='CCDEF', data=anomaly_df)
'''
#anomaly_df['anomaly'] = anomaly_df.apply(lambda x: 1 if (x.name>=pd.to_datetime('2016-02-14')) & (x.name<=pd.to_datetime('2016-02-21')) else 0 , axis=1)
z = anomaly_df.loc['2016-02-14':'2016-02-21', :]

# Remove anomalous dates, to keep normal behaviour of sensors for training
anomaly_df.drop(anomaly_df.loc[(anomaly_df.index>='2016-02-14') & (anomaly_df.index<='2016-02-21')].index, inplace=True)

anomaly_train = anomaly_df['2016-01-01':'2016-09-30']
anomaly_test = anomaly_df['2016-10-01':]

'''
Data provided is already scaled and mssing values have been handled
'''

from sklearn.decomposition import PCA
pca = PCA(n_components=2, svd_solver='full')

anomaly_train_PCA = pca.fit_transform(anomaly_train)

myformat = lambda x: "%.2f" % x
# 0.87 + 0.08 = 0.95
[float(myformat(x)) for x in list(pca.explained_variance_ratio_)]

anomaly_train_PCA = pd.DataFrame(anomaly_train_PCA)
anomaly_train_PCA.index = anomaly_train.index

anomaly_test_PCA = pca.transform(anomaly_test)
anomaly_test_PCA = pd.DataFrame(anomaly_test_PCA)
anomaly_test_PCA.index = anomaly_test.index

''' Rough
plt.scatter(anomaly_train_PCA[0], anomaly_train_PCA[1])
plt.title('PCA Graph')
plt.xlabel('PC1 - {0}%'.format([float(myformat(x)) for x in list(pca.explained_variance_ratio_)][-2]))
plt.ylabel('PC2 - {0}%'.format([float(myformat(x)) for x in list(pca.explained_variance_ratio_)][-1]))

for sample in anomaly_train_PCA:
    plt.annotate(sample, (anomaly_train_PCA[0].loc[sample], anomaly_train_PCA[1].loc[sample]))

plt.show()
'''

def is_pos_def(A):
    if np.allclose(A, A.T):
        try:
            print('____________\t')
            print(np.linalg.cholesky(A))
            print('cholesky above')
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False
    
def cov_matrix(data, verbose=False):
    covariance_matrix = np.cov(data, rowvar=False)
    print('The covariance matrix', covariance_matrix,'\n')
    ## Covariance Matrix, the diagonal values are the same, can be cross checked using function is_pos_def
    if is_pos_def(covariance_matrix):
        ## We need the inverse since we cannot divide using a matrix, but can multiply
        inv_covariance_matrix = np.linalg.inv(covariance_matrix)
        print('\nThe inverse covariance matrix\n', inv_covariance_matrix,'\n')
        if is_pos_def(inv_covariance_matrix):
            return covariance_matrix, inv_covariance_matrix
        else:
            print("Error: Inverse of Covariance Matrix is not positive definite")
    else:
        print("Error: Covariance Matrix is not positive definite")
        
def MahalanobisDist(inv_cov_matrix, mean_distr, data, verbose=False):
    inv_covariance_matrix = inv_cov_matrix
    vars_mean = mean_distr
    diff = data - vars_mean
    md = []
    for i in range(len(diff)):
        md.append(np.sqrt( diff[i].dot(inv_covariance_matrix).dot(diff[i])))
    return md

def MD_detectOutliers(dist, extreme=False, verbose=False):
    ## Function to get outliers
    k = 3. if extreme else 2.
    threshold = np.mean(dist) * k
    outliers = []
    for i in range(len(dist)):
        if dist[i] >= threshold:
            outliers.append(i)  # index of the outlier
    return np.array(outliers)

def MD_threshold(dist, extreme=False, verbose=False):
    k = 3. if extreme else 2.
    threshold = np.mean(dist) * k
    return threshold

data_train = np.array(anomaly_train_PCA.values)
data_test = np.array(anomaly_test_PCA.values)

cov_matrix, inv_cov_matrix = cov_matrix(data_train)
mean_distr = data_train.mean(axis=0)

dist_train = MahalanobisDist(inv_cov_matrix, mean_distr, data_train, verbose=False)
dist_test = MahalanobisDist(inv_cov_matrix, mean_distr, data_test, verbose=False)
threshold = MD_threshold(dist_train, extreme = True)

## visualization distance calculated, should follow chi square distribution
plt.figure()
plt.xlim([0.0, 15])
sns.distplot(np.square(dist_train),
             bins=10,
             kde=False)

## To check is Threshold value calculated is suitable
plt.figure()
plt.xlim([0.0, 5])
plt.xlabel('Mahalanobis distance')
sns.distplot(dist_train,
             bins=10,
             kde=True,
             color='green')

## As seen calc. value 4.0 is large, as per graph resetting threshold to 2.9
threshold_new = 2.9
anomaly_train_final = pd.DataFrame()
anomaly_train_final['M_dist'] = dist_train
anomaly_train_final['Thresh'] = threshold_new
anomaly_train_final['Anomaly'] = anomaly_train_final['M_dist']>anomaly_train_final['Thresh']
anomaly_train_final.index = anomaly_train_PCA.index

anomaly_test_final = pd.DataFrame()
anomaly_test_final['M_dist'] = dist_test
anomaly_test_final['Thresh'] = threshold_new
anomaly_test_final['Anomaly'] = anomaly_test_final['M_dist']>anomaly_test_final['Thresh']
anomaly_test_final.index = anomaly_test_PCA.index

print('\n Anomalous Dates \n\n', anomaly_test_final[anomaly_test_final['Anomaly']==True].index, '\n')


'''These are the dates that display anomalous behaviour, as seen
   -from Dec 16 2016 up until Dec 25 2016 10 days
   -from Dec 27 2016 up until Dec 30 2016 4 days'''

anomaly_all = pd.concat([anomaly_train_final, anomaly_test_final])
anomaly_all.plot(logy=True, figsize=(10, 6), ylim=[1e-1, 1e3], color=['green', 'red'])

'''
## Check with anomalous values reported by The analyst
z_PCA = pca.transform(z)
z_PCA = pd.DataFrame(z_PCA)
z_PCA.index = z.index
z_test = np.array(z_PCA.values)
dist_z_test = MahalanobisDist(inv_cov_matrix, mean_distr, z_test, verbose=False)
z_final = pd.DataFrame()
z_final['M_dist'] = dist_z_test
z_final['Thresh'] = threshold_new
z_final['Anomaly'] = z_final['M_dist']>z_final['Thresh']
z_final.index = z_PCA.index
'''

'''
ref. Articles - 
https://towardsdatascience.com/machine-learning-for-anomaly-detection-and-condition-monitoring-d4614e7de770
https://towardsdatascience.com/how-to-use-machine-learning-for-anomaly-detection-and-condition-monitoring-6742f82900d7
'''
