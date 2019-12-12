import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn import grid_search
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import train_test_split

from nltk.stem.porter import *
stemmer = PorterStemmer()
import re
import random
import nltk

path=r"C:\Users\klest\Desktop\ihu 2nd sem\NLP assignment".replace('\\', '/') #creating the path where we have saved the data sets


df_train =  pd.read_csv(path+"/train.csv", encoding="ISO-8859-1") #importing the given train set with the queries and relevance score

df_pro_desc = pd.read_csv(path+"/product_descriptions.csv", encoding="ISO-8859-1") #importing the given product_descriptions set 

df_attr =  pd.read_csv(path+"/attributes.csv", encoding="ISO-8859-1") #importing the given attributes set 

#creating a new attribute using the brand name of each product
df_brand = df_attr[df_attr.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand"})

#num_train = df_train.shape[0]

df_train = pd.merge(df_train, df_pro_desc, how='left', on='product_uid') #merging train set and product_descriptions set into on common dataframe
df_train = pd.merge(df_train, df_brand, how='left', on='product_uid') #adding the brand name for each of the products taking into cosideration the product id

def str_stem(s): #implenting preprocessing techniques and stemming in a given text s
    if isinstance(s, str): #if s is a subset of str, condition=True
        s = s.lower() #returns the string in lowercase characters
        s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s) #'ungroups' words that are together
#\w only matches the character class [A-Za-z] in bytes patterns,
        s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in. ", s)
    
        s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", s)
    
        s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s)
    
        s = s.replace(" x "," xby ")
        s = s.replace("*"," xby ")
        s = s.replace(" by "," xby")
        s = s.replace("x0"," xby 0")
        s = s.replace("x1"," xby 1")
        s = s.replace("1x","1 xby ")
        s = s.replace("x2"," xby 2")
        s = s.replace("2x","2 xby ")    
        s = s.replace("x3"," xby 3")
        s = s.replace("3x","3 xby ")
        s = s.replace("x4"," xby 4")
        s = s.replace("4x","4 xby ")
        s = s.replace("x5"," xby 5")
        s = s.replace("5x","5 xby ")
        s = s.replace("x6"," xby 6")
        s = s.replace("6x","6 xby ")
        s = s.replace("x7"," xby 7")
        s = s.replace("7x","7 xby ")
        s = s.replace("x8"," xby 8")
        s = s.replace("8x","8 xby ")
        s = s.replace("x9"," xby 9")
        s = s.replace("9x","9 xby ")
        s = s.replace("0x","0 xby ")

        
        s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", s)
    
        s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", s)
        
        s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)
    
        s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", s)
    
        s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", s)
        
        s = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg. ", s)
    
        s = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt. ", s)
        
        s = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt. ", s)
    
        s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", s)
        
        
        
        s = s.replace("whirpool","whirlpool")
        s = s.replace("whirlpoolga", "whirlpool")
        s = s.replace("whirlpoolstainless","whirlpool stainless")

        s = s.replace("  "," ")
        s = (" ").join([stemmer.stem(z) for z in s.split(" ")])
        return s.lower() #returns the processed string in lowercase characters
    else:
        return "null"

def str_common_word(str1, str2):  #finding the number of common words of str1 and str2
    words, cnt = str1.split(), 0
    for word in words:
        if str2.find(word)>=0:
            cnt+=1
    return cnt

def str_whole_word(str1, str2, i_): #searches if str1 exists in str2
    cnt = 0
    while i_ < len(str2):
        i_ = str2.find(str1, i_)
        if i_ == -1:
            return cnt
        else:
            cnt += 1
            i_ += len(str1)
    return cnt

def fmean_squared_error(ground_truth, predictions): #implementing the root mean squared error algorithm
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions)**0.5
    return fmean_squared_error_

df_train['search_term'] = df_train['search_term'].map(lambda x:str_stem(x)) #applying str_stem function for each element of 'search_term' attribute
df_train['product_title'] = df_train['product_title'].map(lambda x:str_stem(x)) #applying str_stem function for each element of 'product_title' attribute
df_train['product_description'] = df_train['product_description'].map(lambda x:str_stem(x)) #applying str_stem function for each element of 'product_description' attribute
df_train['brand'] = df_train['brand'].map(lambda x:str_stem(x)) #applying str_stem function for each element of 'brand' attribute

"""----------------------------------------------------creating new attributes---------------------------------------------------"""

df_train['len_of_query'] = df_train['search_term'].map(lambda x:len(x.split())).astype(np.int64) #calculating the number of words for each query  
df_train['len_of_title'] = df_train['product_title'].map(lambda x:len(x.split())).astype(np.int64)#calculating the title's number of words for each product
df_train['len_of_description'] = df_train['product_description'].map(lambda x:len(x.split())).astype(np.int64) #calculating the description's number of words for each product
df_train['len_of_brand'] = df_train['brand'].map(lambda x:len(x.split())).astype(np.int64) #calculating brand name's number of words for each product

#creating a new attribute that includes the search term, product title and product description
df_train['product_info'] = df_train['search_term']+"\t"+df_train['product_title'] +"\t"+df_train['product_description']

#creating a new attribute by checking if the whole query exists in title
df_train['query_in_title'] = df_train['product_info'].map(lambda x:str_whole_word(x.split('\t')[0],x.split('\t')[1],0))

#creating a new attribute by checking if the whole query exists in product description
df_train['query_in_description'] = df_train['product_info'].map(lambda x:str_whole_word(x.split('\t')[0],x.split('\t')[2],0))

# number of query's words that exists in title
df_train['word_in_title'] = df_train['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))

# number of query's words that exists in product description
df_train['word_in_description'] = df_train['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))

#ratio of the query's words existing in title over the total query's words 
df_train['ratio_title'] = df_train['word_in_title']/df_train['len_of_query'] 

#ratio of the query's words existing in product description over the total query's words 
df_train['ratio_description'] = df_train['word_in_description']/df_train['len_of_query']

#creating a new attribute that includes the search term and the brand
df_train['attr'] = df_train['search_term']+"\t"+df_train['brand']

# number of query's words that exists in brand
df_train['word_in_brand'] = df_train['attr'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))

#ratio of the query's words existing in brand over the total brand's words 
df_train['ratio_brand'] = df_train['word_in_brand']/df_train['len_of_brand']



df_brand = pd.unique(df_train.brand.ravel())

d={}
i = 1
for s in df_brand:
    d[s]=i
    i+=1
df_train['brand_feature'] = df_train['brand'].map(lambda x:d[x])
df_train['search_term_feature'] = df_train['search_term'].map(lambda x:len(x))


#deleting the attributes that we do not need anymore
df_train2= df_train.drop(['search_term','product_title','product_description','product_info','brand','attr'],axis=1)

x=df_train2.drop(['id','relevance'],axis=1).values #deleting id and relevance attributes and storing rest ones in a new Dataframe
y=df_train2['relevance'].values #storing the relevance in a separate Dataframe

#Choosing the classifier we will use for the training of the model
clf = GradientBoostingRegressor()

"""Grid Search implementation"""
from sklearn.model_selection import GridSearchCV
x1_train = np.asarray(df_train2.drop(['id','relevance'],axis=1).values)
y1_train = np.asarray(df_train2['relevance'].values)
y1_train = y1_train.ravel()
grid_search_rf = GridSearchCV(clf, param_grid=dict( ), verbose=3,scoring='mean_squared_error',cv=10).fit(x1_train,y1_train)
print ('best estimator:',grid_search_rf.best_estimator_,'Best Score', grid_search_rf.best_estimator_.score(x1_train,y1_train))
rf_best = grid_search_rf.best_estimator_

n=0

"""Training the model using KFold validation (K=10)"""
from sklearn.model_selection import KFold
kf=KFold(n_splits=10, shuffle=True, random_state=False)
outcomesRf=[]
outcomesRf=[]
for train_id, test_id in kf.split(x,y):
    X_train, X_test = x[train_id], x[test_id]
    y_train, y_test = y[train_id], y[test_id]
    rf_best.fit(X_train,y_train)
    predictions = rf_best.predict(X_test)
    accuracy = fmean_squared_error(y_test, predictions)
    n=n+1
    print(n,accuracy)
    outcomesRf.append(accuracy)
print(np.mean(outcomesRf))

