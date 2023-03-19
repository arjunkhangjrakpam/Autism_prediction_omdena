# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import numpy as np, pandas as pd, os, sklearn, plotly.express as px
from sklearn.pipeline import make_pipeline
from sklearn.utils import check_array
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# To ignore warnings
import warnings
warnings.filterwarnings("ignore")
import os
os.chdir('...\\AUTISM_SCREENING_FOR_TODDLERS\\archive')
df = pd.read_csv('Toddler Autism dataset July 2018.csv')


###################################################################################################
#############################    EXPLORATORY DATA ANALYSIS    #####################################
#############################    EXPLORATORY DATA ANALYSIS    #####################################
#############################    EXPLORATORY DATA ANALYSIS    #####################################
###################################################################################################

df.columns
df.head()
df.rename(columns ={"Class/ASD Traits ":"target"}, inplace=True)
df.head()
df.target = np.where(df.target =='Yes',1,0)
df.head()
yes = df[df['target']==1]
yes.describe()
no = df[df['target']==0]
no.describe()

"""
The minimum value of Qchat-10-Score variable for yes(1) (autistic) is 4.
The maximum value of Qchat-10-Score variable for no(0) (non-autistic) is 3.

"""
df1 = df[['target','Qchat-10-Score']]
df1.describe()

df1.corr()# 81 % correlation

df1_yes = df1[df1['Qchat-10-Score'] >3]
df1_no = df1[df1['Qchat-10-Score'] <=3]

df1_yes.describe()
df1_no.describe()

############################
df1_yes.target.unique()
df1_no.target.unique()
"""
The result shows that for:
    Qchat-10-Score >3 target variable is 1 i.e. patient has autism.
    Qchat-10-Score <=3 target variable is 0 i.e. patient has autism.
    
So by just using this information i.e. with just the Qchat-10-Score variable 
and the cut of value of 3, we can correctly predict whether a toddler has autism 
or not. This is a classical case of data leakage that is the independent 
variable contains information about the target variable hence when ever we train
the model if the train test split is done in such a way that the model recognizes
this information then we will almost everytime get a very high score. We may be
happy that our model is doing so well. In real lif when new dataset comes it may
not contain this information in the Qchat-10-Score variable and hence our model
may not perform well. 

To check this let us :
    1.first train the model with the Qchat-10-Score variable on the toddler 
    dataset and validate our model on the git hub data set. 
    2. secondly train the model without the Qchat-10-Score variable on the
    toddler dataset and validate our model on the git hub data set.
    3. thirdly train the model with the Qchat-10-Score variable on the combined 
    dataset and validate our model on the combined data set. 
    $. fourthly train the model without the Qchat-10-Score variable on the combined 
    dataset and validate our model on the combined data set. 


"""

df1_yes.target.value_counts()
df1_no.target.value_counts()

def data_preprocess(df):
    #Get the new dataset from github
    os.chdir(r'...\AUTISM_SCREENING_FOR_TODDLERS\archive\github_data\Data-Analytics-model-on-Behavioural-Challenges-of-ASD-kids')
    os.listdir()
    dff = pd.read_csv('data_csv.csv')
    dff['Sex'] = np.where(dff['Sex']=='F','f','m')
    dff1 = dff[[ 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8','A9', 'A10_Autism_Spectrum_Quotient','Age_Years', 'Qchat_10_Score',  'Sex', 'Ethnicity', 'Jaundice','Family_mem_with_ASD', 'Who_completed_the_test', 'ASD_traits']]
    #max age in toddlers dataset is 36 months i.e. 3 years
    df.Age_Mons.max()
    #filter from the new data all records with 'Age_Years' <= 3
    dff2 = dff1[dff1['Age_Years']<=3]
    yes = dff2[dff2['ASD_traits']=='Yes']
    yes.describe()
    no = dff2[dff2['ASD_traits']=='No']
    a = df[['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10','Age_Mons', 'Qchat-10-Score', 'Sex', 'Ethnicity', 'Jaundice', 'Family_mem_with_ASD', 'Who completed the test', 'target']]
    b = dff2
    b.columns
    b.rename(columns={'A10_Autism_Spectrum_Quotient':'A10'},inplace = True)
    b.rename(columns = {'ASD_traits':'target'},inplace = True)
    b['Age_Mons']=b.Age_Years*12
    #b.drop(['Qchat_10_Score','Age_Years'],axis=1,inplace = True)
    b.drop(['Age_Years'],axis=1,inplace = True)
    b.target = np.where(b['target'] == 'Yes',1,0)
    a.rename(columns = {'Who completed the test':'Who_completed_the_test'},inplace = True)
    a.rename(columns = {'Qchat-10-Score':'Qchat_10_Score'},inplace = True)
    b = b[['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'Age_Mons','Qchat_10_Score','Sex', 'Ethnicity', 'Jaundice', 'Family_mem_with_ASD','Who_completed_the_test', 'target']]
    b.columns
    a.columns
    b['Qchat_10_Score'] = b['Qchat_10_Score'].fillna(0).astype(np.int64)
    a['dataset']='toddler'
    b['dataset']='github'
    c=a.append(b)
    ### save the preprocessed files
    a.to_csv(r'...\AUTISM_SCREENING_FOR_TODDLERS\data\original_data.csv',index = False)
    b.to_csv(r'...\AUTISM_SCREENING_FOR_TODDLERS\data\github_data.csv',index = False)
    c.to_csv(r'...\AUTISM_SCREENING_FOR_TODDLERS\data\combined_data.csv',index = False)

#data_preprocess(df)

################################################ Read the files saved from before ###########################
aa = pd.read_csv(r'...\AUTISM_SCREENING_FOR_TODDLERS\data\original_data.csv')
bb = pd.read_csv(r'...\AUTISM_SCREENING_FOR_TODDLERS\data\github_data.csv')
cc = pd.read_csv(r'...\AUTISM_SCREENING_FOR_TODDLERS\data\combined_data.csv')


aa.columns
bb.columns
cc.columns

## Create dummy variable for categorical variables

def get_dummy(df):
    #Introducing dummy variables for all categorical variables by dropping the first dummy variable
    Sex = pd.get_dummies(df.Sex, prefix='Sex', drop_first=True)
    Ethnicity = pd.get_dummies(df.Ethnicity, prefix='Ethnicity', drop_first=True)
    Jaundice = pd.get_dummies(df.Jaundice, prefix='Jaundice', drop_first=True)
    Family_mem_with_ASD = pd.get_dummies(df.Family_mem_with_ASD, prefix='Family_mem_with_ASD', drop_first=True)
    Who_completed_the_test = pd.get_dummies(df["Who_completed_the_test"], prefix='Who_completed_the_test', drop_first=True)
    #Introducing dummy variables for all categorical variables by dropping the first dummy variable
    df.drop(["Sex","Ethnicity","Jaundice","Family_mem_with_ASD","Who_completed_the_test"], axis = 1,inplace=True)
    df =  pd.concat([df, Sex,Ethnicity,Jaundice,Family_mem_with_ASD,Who_completed_the_test ], axis=1)
    return(df)

def save_files():
    aaa = get_dummy(aa)
    bbb = get_dummy(bb)
    ccc = get_dummy(cc)
    aaa.columns
    bbb.columns
    ccc.columns

    aaa.to_csv(r'...\AUTISM_SCREENING_FOR_TODDLERS\data\original_data_one_hot_encoded.csv',index = False)
    bbb.to_csv(r'...\AUTISM_SCREENING_FOR_TODDLERS\data\github_data_one_hot_encoded.csv',index = False)
    ccc.to_csv(r'...\AUTISM_SCREENING_FOR_TODDLERS\data\combined_data_one_hot_encoded.csv',index = False)

#save_files()

###########################################################################################################################
################################################        END OF EXPLORATORY DATA ANALYSIS       ###########################
################################################        END OF EXPLORATORY DATA ANALYSIS       ###########################
################################################        END OF EXPLORATORY DATA ANALYSIS       ###########################
###########################################################################################################################

#------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------


################################################ Read the files saved one hot encoded files  ###########################
aa = pd.read_csv(r'...\AUTISM_SCREENING_FOR_TODDLERS\data\original_data_one_hot_encoded.csv')
bb = pd.read_csv(r'...\AUTISM_SCREENING_FOR_TODDLERS\data\github_data_one_hot_encoded.csv')
cc = pd.read_csv(r'...\AUTISM_SCREENING_FOR_TODDLERS\data\combined_data_one_hot_encoded.csv')


aa.columns
bb.columns
cc.columns

aa.shape # (1054, 30)
bb.shape # (147, 32)
cc.shape # (147, 32)

## define functions to train the models
def train_model1(df1,test_size):
    df1.drop('dataset',axis = 1, inplace=True)
    # Putting feature variable to X
    X = df1.drop(['target'], axis=1)
    # Puttting response variable to y
    y = df1.loc[:,['target']]

    # Splitting the data into train and test with test size as 30% and random state as 101

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size= test_size)
    # Pipeline Estimator 
    standardscaler =StandardScaler()
    radomforestclassifier = RandomForestClassifier(n_jobs = -1,verbose = 0)
    pipeline = make_pipeline(standardscaler,radomforestclassifier)
    # fit model on training data
    pipeline.fit(X_train,y_train)


    # Predict the sales of the test data
    y_test['pred'] = pipeline.predict(X_test)
    from sklearn import metrics
    # testing score
    score = metrics.f1_score(y_test['target'], y_test['pred'],labels=None, pos_label=1)

    print("F1 score for test data is : ",score)
    print("Accuracy score for test data is : ",metrics.accuracy_score(y_test['target'], y_test['pred']))
    print('train_test_split_ratio is : ', test_size)


def train_model2(df1,df2):
    # Putting feature variable to X
    X_train = df1.drop(['target'], axis=1)
    # Puttting response variable to y
    y_train = df1.loc[:,['target']]


    # Putting feature variable to X
    X_test = df2.drop(['target'], axis=1)
    # Puttting response variable to y
    y_test = df2.loc[:,['target']]


    standardscaler =StandardScaler()
    radomforestclassifier = RandomForestClassifier(n_jobs = -1,verbose = 0)
    pipeline = make_pipeline(standardscaler,radomforestclassifier)
    # fit model on training data
    pipeline.fit(X_train,y_train)

    #Test data is the github dataset
    X_test.columns


    # Predict the sales of the test data
    y_test['pred'] = pipeline.predict(X_test)
    from sklearn import metrics
    # testing score
    score = metrics.f1_score(y_test['target'], y_test['pred'],labels=None, pos_label=1)

    print("F1 score for test data is : ",score)
    print("Accuracy score for test data is : ",metrics.accuracy_score(y_test['target'], y_test['pred']))



###########################################################################################################################
# 1. First train the model with the Qchat-10-Score variable on the toddler 
#    dataset and validate our model on the test data of the toddler data set. 
###########################################################################################################################
df1 = pd.read_csv(r'...\AUTISM_SCREENING_FOR_TODDLERS\data\original_data_one_hot_encoded.csv')
train_model1(df1, test_size=0.25)

#Result
"""
F1 score for test data is :  1.0
Accuracy score for test data is :  1.0
"""

###########################################################################################################################
# 2. Secondly train the model with the Qchat-10-Score variable on the toddler 
#    dataset and validate our model on the git hub data set. 
###########################################################################################################################

df =  pd.read_csv(r'...\AUTISM_SCREENING_FOR_TODDLERS\data\combined_data_one_hot_encoded.csv')
df1 = df[df['dataset']=='toddler']
df2 = df[df['dataset']=='github']
df1.drop('dataset',axis = 1, inplace=True)
df2.drop('dataset',axis = 1, inplace=True)

train_model2(df1, df2)
#Result
"""
F1 score for test data is :  0.8488372093023255
Accuracy score for test data is :  0.8231292517006803
"""
###########################################################################################################################
# 3. Thirdly train the model without the Qchat-10-Score variable on the toddler 
#    dataset and validate our model on the git hub data set. 
###########################################################################################################################

df =  pd.read_csv(r'...\AUTISM_SCREENING_FOR_TODDLERS\data\combined_data_one_hot_encoded.csv')
df.drop('Qchat_10_Score',axis = 1, inplace=True)
df1 = df[df['dataset']=='toddler']
df2 = df[df['dataset']=='github']
df1.drop('dataset',axis = 1, inplace=True)
df2.drop('dataset',axis = 1, inplace=True)

train_model2(df1, df2)
# Result
"""
F1 score for test data is :  0.9466666666666668
Accuracy score for test data is :  0.9455782312925171
"""
###########################################################################################################################
# 4. Fourthly train the model with the Qchat-10-Score variable on the combined 
#    dataset and validate our model on the combined data set. 
###########################################################################################################################

for i in range(10,50):
    df1 =  pd.read_csv(r'...\AUTISM_SCREENING_FOR_TODDLERS\data\combined_data_one_hot_encoded.csv')
    train_model1(df1, test_size=i*0.01)
    print(" Value of i is :", i)


###########################################################################################################################
# 5. Fifth train the model without the Qchat-10-Score variable on the combined 
#    dataset and validate our model on the combined data set. 
###########################################################################################################################

for i in range(10,50):
    df1 =  pd.read_csv(r'...\AUTISM_SCREENING_FOR_TODDLERS\data\combined_data_one_hot_encoded.csv')
    df1.drop('Qchat_10_Score',axis = 1, inplace=True)
    train_model1(df1, test_size=i*0.01)
    print(" Value of i is :", i)
    
###########################################################################################################33


cc =  pd.read_csv(r'...\AUTISM_SCREENING_FOR_TODDLERS\data\combined_data_one_hot_encoded.csv')



cc.columns

df = cc[['Qchat_10_Score','target']]
df_1 = df[df['target']==1]
df_0 = df[df['target']==0]

df_1.describe()
df_0.describe()
cc.target.value_counts()

df_1['Qchat_10_Score'].value_counts()
df_0['Qchat_10_Score'].value_counts()


#################################################################

### Further actions to investigate.
# Remove outliers from 'Qchat_10_Score' variable for both yes(1) and no (0).
# Read the criteria for the scoring 'Qchat_10_Score' variable in both the datasets and see if we can do any further data cleaning.