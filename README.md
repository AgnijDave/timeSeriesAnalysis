# timeSeriesAnalysis
Anomaly Detection in sensor readings of an Offshore Oil &amp; Gas Company

Given data is already pre-processed (missing values handled) and scaled, Analyst has manually cross checked first 9 months and the data for the last 3 months needs to be analysed and predicted upon.
Anamolus dates identified by Analyst in the 1st 9 months not used in training.

Approach - 
Use PCA to decompress the data into 2 PC's
These PC's are used to calculate Mahalanobis distance.

Reason for using this distance metric-
1) Euclidean distance assumption - no correlation amongst variables.
2) Mahalanobis uses the covariance matrix in calculation in order to forgeo this assumption.

5 Sensors Data over a period of 1 year 2016

   Figure -      <img width="362" alt="times" src="https://user-images.githubusercontent.com/46378477/131498072-37729a8b-7a38-4cd4-828b-2a61a35605bb.PNG">
