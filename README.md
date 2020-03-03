# Machine-Learning-ALF

Machine Learning 
Submitted by Harmeet Chhibber


Background Information:
The main objective of this project is to predict liver failure using demographic and personal health information. About 8785 individuals who were 20 years and older and recorded their data in 27 variables were treated as the training data. This information was collected using direct interviews, examinations and blood samples. Data collected from surveys in 2008-09 and 2014-15 at JPAC Centre for Health Diagnosis and Control in India.
Preprocessing and approach:
I started off with preprocessing the dataset to get the data ready for applying the models. I got rid of the null values in the response variables. 
The next step was to impute the missing values in each of the features in the dataset. Roughly 6000 instances remained with response variables out of 8500. For each of the given features, median seemed like the best selection of imputation value since there were some categorical, some continuous variable. Also, within the categorical variables there were ordinal and nominal type, however none of the variables had missing values summing up to more than 100 which meant that imputed values would not make a huge difference but deleting the rows did not make a lot of sense as we only have 500 positive label values and about 5500 negative values. Although the data set is imbalanced but there is no more data available to get rid of this situation. Also, since the analysis is about a health issue related problem we have to account for the false negatives and the false positives too. We must consider the model with the least number of the two. Recall and precision along with accuracy is the most desirable step to proceed with in order to test the model performance and decide upon which of the selected models is the most appropriate to be used for computing the conclusion. 
Experiment Results and Discussion:
Since our dataset has binary response variable given based on mulitple accounted patient attributes, using a classification algorithm is the approach. During this semester we have concentrated on the working of classification models based on supervised learning techniques. I would like to test the performance of both parametric and non-parametric models and different generative and discriminative models. Based on that I have selected to start off with Decision Trees since they take any kind of features and normalization or standardization is not required for predicting the results. Following that I will be using random forest to compute the accuracy. Random forest uses multiple decision trees combines them on the basis of ensemble learning method of bagging and resamples the dataset in many smaller datasets and combines them to estimate the outcomes of the final classifier and outputs the response. The next few models required to standardize the data and scale down to the same degree. Naïve Bayes which is a generative learning algorithm. The next one is KNN which is an instance-based model and works with the entire training data to classify the new test instances. Following that SVM and Logistic regression which are discriminative classifiers which will help us learn a decision boundary based on the training sets and then predict based on the decision boundary.  
Methods, Results and Evaluation:
ClassificationModel	Accuracy	Recall	Precision
Decision tree	        88.80%	43.43%	35.24%
Random Forest 	      92.53%	13.13%	76.47%
Naïve Bayes	          85.24%	59.6%	30%
K- Nearest Neighbors	91.29%	12.12%	40%
SVM (linear)	        91.45%	21.21%	45.65%
SVM (kernel trick)	  92.78%	23.23%	67.65%
Logistic Regression	  92.37%	14.14%	66.67%

Conclusions:
After estimating the performance of all the models and comparing the metrics, given the data we have, it is prominent that the accuracy is not the best measure for concluding to the best model for our prediction on future cases. Recall and precision would be the most accurate measurements as we have a problem at hand with which we can not afford to have a lot of false negative and false positives are expensive too. Our optimal model should be a good estimator of each of these metrics. 
Checking the table above, even though the random forest classifier has the best accuracy, the recall is very low. Precision for the random forest is not too bad but we want our model to be decently performing for each of the metrics stated above. Taking that in account, Naïve Bayes performs the best as the accuracy is 85%, recall is almost 60% and precision is 30%. 
Recall takes false negatives in account and are inversely proportional to the recall, so as the recall increases the false negatives decrease. For any health-related data, reducing the false negatives is the first priority. We do not want to classify an infected person as healthy as they will suffer from that. 
Precision on the other hand takes false positives in account. Which can be ignored given the situation that we have an imbalanced dataset and then solution of collecting or adding more data points or instances is not viable at this stage. 

Out of all the other models, SVM has the best accuracy, but accuracy is not the best evaluating metric for our data problem. So, we will look at the next best metric. Precision wise the best estimator is random forest model. But since we already have checked that Recall is the best estimator of the model importance in our case, we will be treating Naïve Bayes as our best model and using it to classify all the further test cases. In case, we were able to collect more data for the positive class, we will be able to build more robust model and we might get better scores for the above metrics from the other models too. 
To conclude, I would like to state that Naïve Bayes is the best predictor model given our data set as all the information we have in order to train the model and predict future cases. 
