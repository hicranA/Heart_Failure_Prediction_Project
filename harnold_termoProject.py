#!/usr/bin/env python
# coding: utf-8

# ### Auther : Hicran Arnold <br>MET CS 677 Term Project <br> Title: Heart Failure Prediction 
# Using Logistic Regression Machine Learning Algorithms<br> Date:04/27/2022

# ### Table Of Contents 
# **1. Abstract**<br>
# **2. Introduction**<br>
# **3. Python Libraries**<br>
# **4. Exploring and Cleaning the Data**<br>
# **5. Transforming the Categorical Variables: Creatinging Dummy Variables**<br>
# **6. Split Training and Test Datasets**<br>
# **7- Creating Logistic Regression Model and Fitting our Data**<br>
# **9- Feature Selection Model Improvement**<br>
# **9-Final Model with Selected Parameters**<br>
# **10- Evaluate the Model**<br>



# ### 2- INTRODUCTION
# **2.1 Information About The Data Set**
# - Original Data Set and Source Info: [ Retrieved from fedesoriano. 
# Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction?select=heart.csv) 
# - 12 Attributes
# - 918 observations 
# 
# **2.2 Atribute(feature) Information**
#   - Age: age of the patient [years]
#   - Sex: sex of the patient [M: Male, F: Female]
#   - ChestPainType: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
#   - RestingBP: resting blood pressure [mm Hg]
#   - Cholesterol: serum cholesterol [mm/dl]
#   - FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
#   - RestingECG: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality 
# (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
#   - MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]
#   - ExerciseAngina: exercise-induced angina [Y: Yes, N: No]
#   - Oldpeak: oldpeak = ST [Numeric value measured in depression]
#   - ST_Slope: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
#   - HeartDisease: output class [1: heart disease, 0: Normal]

# ---

# ### 3- Import Import Python Libraries
# Below are the list of python libraries that we need for this project

#
# for data processing
import pandas as pd
import numpy as np
# for vizulazations
import matplotlib.pyplot as plt
import seaborn as sns
# scalling the data for the machine learning model
from sklearn.preprocessing import StandardScaler
# machine learning libraries for data modeling
from  sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,accuracy_score,plot_roc_curve
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import log_loss, roc_auc_score, recall_score, precision_score, average_precision_score, f1_score, classification_report, accuracy_score, plot_roc_curve, plot_precision_recall_curve, plot_confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics


# ---

# ### 4. Exploring and Cleaning the Data

# **4.1 Loading the raw data**


data_heart = pd.read_csv("heart.csv")
print("onservation size: ", len(data_heart))
print(data_heart.head())


# **4.2 Remove duplicate or irrelevant observations** <br>
# In this step I will be checking to see if there are any dublicates or irrelavant observations.
# In our data set we have no dublicates.


# this checks to see if df contains any dublicates
print(data_heart.duplicated().sum() )


# **4.3 Checking for missing data**

# Checking for null and NA values: We have no null values. We got zero for each features. 


print(data_heart.info())


# **4.4 Investigating all the elements whithin each Feature**
# In this section, we will check the data to see if there is any variable that looks different than what is listed on the data source website. I did not see any abnormality. 


feature_no =0
for column in data_heart:
    unique_values = np.unique(data_heart[column])
    nr_values = len(unique_values)# number of unique_values
    feature_no+=1 # countin feature to make sure we screen all of the features
    print("Feature: ",feature_no, column)
    if nr_values <= 10: # if the number of values are less than 10 
        print("The number of unquie values for feature {} is: {} -- {}".format(column, nr_values, unique_values))
    else:
        print("The number of unquie values for feature {} is: {}".format(column, nr_values)) # only print the number of values
    print("*********************************")


# ### **4.5 Summarizing and Vizulation Data** <br>
# for this step we need split our data two as categorical and numerical



categ_data =[]
numeric_data = []
binary_data = []
for column in data_heart:
    unique_values = np.unique(data_heart[column])
    nr_values = len(unique_values)
    if data_heart.dtypes[column] == "object":
        categ_data.append(column)
    else:
        if nr_values == 2:
            binary_data.append(column)
        else:
            numeric_data.append(column)
print("categorical data")    
print(categ_data)
print("numeric data")
print(numeric_data)
print("binary data ( numeric 0 and 1)")
print(binary_data)


# **4.5a numerical data summary**

# Now we can vizulazi our numeric data and check to see if there is 
# any skweness and any linear relationship in between our features. 
# Below plot indicates that we do not have any high correlation in our data. 
# However we see some skewnees in oldpeak and chelestrol


p = sns.pairplot(data_heart.loc[ : ,numeric_data], corner=True)
plt.gcf().set_size_inches(15, 8)
plt.show()

# The below pairplot show us values by heart disase feature


data_comparison = data_heart.loc[ : ,numeric_data]
data_comparison["HeartDisease"]= data_heart["HeartDisease"]
g = sns.PairGrid(data_comparison, hue="HeartDisease" , corner=True, palette = "mako")
g.map_diag(sns.histplot, multiple="stack", element="step", color=".3", palette = "mako")
g.map_offdiag(sns.scatterplot)
g.add_legend()
plt.gcf().set_size_inches(15, 8)
plt.show()

# we do not see high correlation in between variables


# correlation matrix
data_comparison = data_heart.loc[ : ,numeric_data]
data_comparison["HeartDisease"]= data_heart["HeartDisease"]
matrix = data_heart.loc[ : ,numeric_data].corr().round(3)
mask = np.triu(np.ones_like(matrix, dtype=bool))
sns.heatmap(data_heart.loc[ : ,numeric_data].corr(),annot=True, vmax=1, vmin=-1, center=0, cmap='PiYG', mask=mask)
plt.show()


# **4.5b Outliers Detection** <br>
# I noticed some outliers, these outliers needs further investigation to see if we need to remove them.




sns.boxplot(x=data_heart.Cholesterol)



data_heart["Cholesterol"].hist()




from scipy import stats
z = np.abs(stats.zscore(data_heart["Cholesterol"]))

threshold = 3
print("##########")
print(np.where(z > 3))



sns.boxplot(x=data_heart.Oldpeak)




from scipy import stats
z_2 = np.abs(stats.zscore(data_heart["Oldpeak"]))

threshold = 3
print("##########")
print(np.where(z_2 > 3))




q_low = data_heart["Oldpeak"].quantile(0.01)
q_hi  = data_heart["Oldpeak"].quantile(0.99)

df_filtered = data_heart[(data_heart["Oldpeak"] < q_hi) & (data_heart["Oldpeak"] > q_low)]
print(q_hi)


# **4.5c Categorical Data Summary**

# reviwing the categoerical data
# - we see that our data has more patiant that does not have heart failure so when we are creating our model we need to split accordingly so that we do not have bias.
# - We can identify some of the classifications by just reviewing the plots below, for example, 
# there is not a clear difference in restingesg in between heart disease and not hear disease butin the ST_slope: 
# we see a big number of patient who has  heart desiase have a ST slope flat because of the difference you can easily tell.


ax = sns.countplot(x="HeartDisease", data=data_heart,palette = "mako")
for container in ax.containers:
        ax.bar_label(container)
    
plt.show()




ax = sns.countplot(x="HeartDisease", data=data_heart,palette = "mako")
for container in ax.containers:
        ax.bar_label(container)
    
plt.show()



# we need to combine categorical data and binary data to review 
category_data_joined = categ_data + binary_data
print(category_data_joined)
for f in categ_data:
    ax = sns.countplot(x = f, data = data_heart.loc[ : ,category_data_joined], hue = 'HeartDisease', palette = "mako")
    for container in ax.containers:
        ax.bar_label(container)
    
    plt.show()


# ---

# ### **5. Transforming the Categorical Variables: Creatinging Dummy Variables** <br>
# In this section we will transfer our categorical data to dummy variables. We already have some binary data we just need to transform the rest of it. 
# I did not want to use to pandas dummy variable builtin function because of the "dummy variable trap". 
# It made my code very messy therefore I decided to transfrom them one by using my own function.


print("we need to transform these below three columns", categ_data)


def transformCatData(colname, data):
    le = LabelEncoder()

    def transformCate(colname, df):
        new_col_name = colname+"_nu"
        df_new = pd.DataFrame(le.fit_transform(df[colname].values), columns =[new_col_name])
        merged= pd.concat([df, df_new], axis="columns")
        del merged[colname]
        return merged

    for i in range(len(categ_data)):
        if i == 0:
            data_transformed= transformCate(colname=categ_data[i] , df= data)
        else:
            data_transformed= transformCate(colname=categ_data[i],df= data_transformed)
    return  data_transformed        
        
data_transformed =  transformCatData(colname=categ_data,data= data_heart) 
print("original order", data_transformed.columns)
# changing the order
data_transformed =  data_transformed[['Age',
 'RestingBP',
 'Cholesterol',
 'FastingBS',
 'MaxHR',
 'Oldpeak',
 'Sex_nu',
 'ChestPainType_nu',
 'RestingECG_nu',
 'ExerciseAngina_nu',
 'ST_Slope_nu',
 'HeartDisease',]]

print(len(data_heart.columns.tolist()))
print("new order, ", data_heart.columns.tolist())


# ---

# ### **6-Split Training and Test Datasets**



x = data_transformed.iloc[ : , :11]
print("**** x ***")
#print(x.head())
y=data_transformed.loc[ : ,"HeartDisease"]
print("**** y***")
#print(y.head())

x_train,x_test,y_train, y_test = train_test_split(x, y, 
                                                  test_size=0.5,
                                                  random_state=11, 
                                                  stratify=y, shuffle=True)


print(x_train.head())


# ---

# ### **7- Creating Logistic Regression Model and Fitting our Data**<br>



log_reg_classifier = LogisticRegression(solver='lbfgs', max_iter=1000) # intiate the log reg
log_reg_classifier.fit(x_train,y_train)# fit the training data to train the data
y_pred = log_reg_classifier.predict(x_test) # use test data to predict
accuracy = accuracy_score(y_test, y_pred)
accuracy


# Since I did a lot of manual data preparation. I want to check to see if I can use create a piplene to do all these manual steps
# I also wanted to compare my results so that if I made any mistake i should have two very different accuracy result.

# in the below code I used sklearn processor pipline. I created new data split fromt the original data.
# This pipeline eliminates dummy variable process

# ---

# ### **8- Feature Selection**<br>
# Before evaluating our model I want evaluate the some of the features that are not very important for my model.
# first we check coef values of each features and plot

#


def dropFeatureAccr(x_train,x_test, y_train, y_test):
    accuracy_featuresDrop = []
    column_name= list(x_train.columns.values.tolist())
    for column in range(len(column_name)):
        x_train_modified = x_train.drop(columns=[column_name[column]])
        x_test_modified = x_test.drop(columns=[column_name[column]])
        log_reg_classifier = LogisticRegression(solver='lbfgs', max_iter=1000)
        log_reg_classifier.fit(x_train_modified,y_train)
        y_pred_mod = log_reg_classifier.predict(x_test_modified)
        accuracy_featuresDrop.append(accuracy_score(y_test,y_pred_mod))      
    dictionary_feature_drop = dict(zip(column_name, accuracy_featuresDrop))        
    return dictionary_feature_drop

logisticRegDrop =dropFeatureAccr(x_train= x_train,y_train= y_train, x_test= x_test, y_test = y_test) 
print(logisticRegDrop)
print("max effect ")
print("feature", max(logisticRegDrop, key=logisticRegDrop.get))
print("accuracy score ", logisticRegDrop.get(max(logisticRegDrop, key=logisticRegDrop.get)))
print("when we drop it the accurasy score went down effect")
print("feature",min(logisticRegDrop, key=logisticRegDrop.get))
print("accuracy score ", logisticRegDrop.get(min(logisticRegDrop, key=logisticRegDrop.get)))


lists = logisticRegDrop.items() # sorted by key, return a list of tuples

x, y = zip(*lists) # unpack a list of pairs into two tuples

plt.plot(x, y)
plt.xticks(x,[0,1,2,3,4,5,6,7,8,9,10])
plt.show()




n = pd.DataFrame.from_dict(logisticRegDrop,orient='index' ).sort_values(by=0)
print(n)

# we see in our graph whem we drop our **MaxHR** we see that our accuracy goes up
# we also see that  our most important our value is the **ST_Slope** , we need to make sure that we keep this value in our model and we can also do a more research on the ST_Slope too

# we will work on the train data set 
# we will split our data set to two ( one with the Heart Disease and with no heart disease to compare them)
# first we will examine the numeric data and then not numeric data

# ### **9-Final Model with Selected Parameters**<br>



x_train_modified = x_train.drop(columns=["MaxHR", "ChestPainType_nu", "RestingECG_nu"])
x_test_modified = x_test.drop(columns=["MaxHR", "ChestPainType_nu", "RestingECG_nu"])
log_reg_classifier.fit(x_train_modified,y_train)# fit the training data to train the data
y_pred_mod = log_reg_classifier.predict(x_test_modified) # use test data to predict
accuracy = accuracy_score(y_test, y_pred_mod)
print(accuracy)


# ---

# ### **10- Evaluate the Model**<br>



fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_mod)
auc = metrics.roc_auc_score(y_test, y_pred_mod)
#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()


# The Area Under the Curve (AUC) is high therefore our model did well. 



print('Log loss = {:.5f}'.format(log_loss(y_test, y_pred_mod)))
print('AUC = {:.5f}'.format(roc_auc_score(y_test, y_pred_mod)))
print('Average Precision = {:.5f}'.format(average_precision_score(y_test, y_pred_mod)))
print('\nUsing 0.5 as threshold:')
print('Accuracy = {:.5f}'.format(accuracy_score(y_test, y_pred_mod)))
print('Precision = {:.5f}'.format(precision_score(y_test, y_pred)))
print('Recall = {:.5f}'.format(recall_score(y_test, y_pred_mod)))

print('F1 score = {:.5f}'.format(f1_score(y_test, y_pred_mod)))

print('\nClassification Report')
print(classification_report(y_test, y_pred_mod))
sse =np.sum((y_pred_mod - y_test)**2)
print("Sum of squared error",sse)




cm = confusion_matrix(y_test, y_pred_mod)
TN_log, FP_log, FN_log, TP_log = cm.ravel()
print(TN_log, FP_log, FN_log, TP_log)
group_names = ["TN","FP","FN","TP"]
group_counts = ["{0:0.0f}".format(value) for value in
                cm.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     cm.flatten()/np.sum(cm)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cm, annot=labels, fmt='', cmap='mako')


#

TPR_log = TP_log/(TP_log + FN_log)
TNR_log = TN_log /(TN_log + FP_log)

my_accuracy_score_log= accuracy_score(y_test, y_pred_mod)
my_matrix_values_1 = {"TN":TN_log, "FP":FP_log, "FN":FN_log, "TP":TP_log,"Accuracy":my_accuracy_score_log,
                      "TPR":TPR_log,"TNR":TNR_log}
result = pd.DataFrame(my_matrix_values_1 , index=[0])
print(result)


# our TNR and TPR scores are very close, also our over all accuracy is very close. Over all our model did 
# a pretty good job but we can still improve our model by trying some other technics. 

# Our TNR and TPR scores are very close, also our overall accuracy is very close. Overall our model did 
# a pretty good job but we can still improve our model by trying some other technics, like removing some outliers or eliminating more irrelavant features.
# I also want to show that we can also add our prediction for each data points. Please see example below. 



test_prob = log_reg_classifier.predict_proba(x_test_modified)[:, 1] # probabilty
test_pred = log_reg_classifier.predict(x_test_modified)#target = values
print(test_prob[0])
print(test_pred[0])


# the below results gives us the odd ratio for our each feature. For example sex, for male (sex=1) 
# to the odds of having heart disease for females is exp(1.215323). You can derive it based on the logistic regression equation.




coefficients = np.hstack((log_reg_classifier.intercept_, log_reg_classifier.coef_[0]))
print(len(coefficients))
print(x_train.columns.values.tolist())
print(pd.DataFrame(data={'variable': ['intercept'] + x_train_modified.columns.values.tolist(), 'coefficient': coefficients}))

