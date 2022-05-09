#!/usr/bin/env python
# coding: utf-8

# # Predicting Used Cars Prices
# 
# This data is scraped every few months, it contains most all relevant information that **Craigslist** provides on car sales including columns like price, condition, manufacturer, latitude/longitude, and 18 other categories. 
# 
# For this notebook I will only import relevant informations, since entries id or urls and car's vehicle identification number etc. won't be needed for this notebook. 
# 
# The Dataset can be found [here](https://www.kaggle.com/austinreese/craigslist-carstrucks-data)

# ## Loading Packages and Dataset

# In[68]:


import pandas as pd
import numpy as np
from numpy import mean

import geopandas
# import plotly.offline as pyo
# import plotly.graph_objs as go
# # Set notebook mode to work in offline
# pyo.init_notebook_mode()
import plotly.offline as pyo
pyo.init_notebook_mode(connected=False)
from plotly.offline import iplot

import plotly.express as px
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt
import seaborn as sns

from tabulate import tabulate

from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, KFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn import metrics
import math


# In[2]:


# Importing only relevant columns, since Informations such as entries url or id are not relevant for this analysis. 

col_list = ['region', 'state', 'price', 'year', 'manufacturer', 'model', 'condition',
            'cylinders', 'fuel', 'odometer', 'title_status', 'transmission', "lat", "long",
            'drive', 'type']

schema = {'region':"category", 'state':"category",'price':"int64", 'year':"float32",  'manufacturer':"category", 'model':"str", 
          'condition':"category",'cylinders':"category", 'fuel':"category", 'odometer':"float32", 'title_status':"category", 
          'transmission':"category", 'drive':"category", 'type':"category"}

df_initial = pd.read_csv("./vehicles.csv", usecols=col_list, dtype=schema)


# In[3]:


df_initial.info()


# In[4]:


df_initial.sample()


# In[5]:


df_initial.shape


# ## Preprocessing and cleaning the Data

# In[6]:


def missing_values(df):
    return (df.isnull().sum()*100 /len(df)).sort_values(ascending=False)
    


# In[7]:


df1 = df_initial.copy(deep=True)


# In[8]:


df1.head(10)


# In[9]:


missing_values(df1)


# In[10]:


df1.shape


# In[11]:


df1.describe([0.01, 0.25, 0.5, 0.75, 0.99])


# ### First look

# In[12]:


plt.figure(figsize = (15,8))
ax = sns.boxplot(x=df1["price"], orient="h")
ax.set_xscale('log')


# In[13]:


plt.figure(figsize = (15,8))
ax = sns.boxplot(x=df1["odometer"], orient="h")
ax.set_xscale('log')


# In[14]:


df1[df1["price"] == 0]


# In[15]:


df1[df1["price"] < 10].shape


# In[16]:


df1[df1["price"] < 100].shape


# In[17]:


# Since the dataset does not give an explain of why some cars are listed for zero and the cars seem to be in good shape, I will
# drop all these entries plus all entries with prices less than 100, because they could bias the results. 
print(" Number of cars which are listed for less than $100 is {} however the majority {} are listed for less than $10 \n {} Cars are listed for more than $100k"
      .format(len(df1[df1.price <100]), len(df1[df1.price <10]),
              len(df1[df1.price >100000])
             )
     ) 
print("\n Cars with odometers values larger than 300k", len(df1[df1.odometer >300000]))


# In[18]:


to_drop = df1[(df1.price <100)|(df1.price >100000)|(df1.odometer >300000)].index


# In[19]:


print("Dropping %.2f percent of the entries to eliminate outliers" %((len(to_drop)/df1.shape[0])*100))


# In[20]:


df1.drop(to_drop, axis=0, inplace=True)
df1.dropna(subset = ["model", "year", "odometer"], inplace=True)


# In[21]:


df1.shape


# In[22]:


def models_dist(df):
    
    models = df.groupby("model")["region"].count().reset_index()                     .rename(columns={"region":"count"}).sort_values("count")

    models["quintile"] = 1/len(models)*100
    models["quintile (cum)"] = models["quintile"].cumsum()

    models["share of total"] = (models["count"]/sum(models["count"]))*100
    models["share of total (cum)"] = models["share of total"].cumsum()

    models.drop(columns=["quintile", "share of total"], inplace=True)
    return models


# In[23]:


def models_plot(df):
    plt.figure(figsize = (15,8))
    plt.plot( df["quintile (cum)"], df["share of total (cum)"])
    plt.fill_between(df["quintile (cum)"], df["share of total (cum)"])
    plt.title('Distribution of car models', fontsize=20)
    plt.ylabel('Cummulative Share', fontsize=15)
    plt.xlabel('Model Quintiles', fontsize=15)
    plt.show()


# In[24]:


models_plot(models_dist(df1))


# In[25]:


models = df1.groupby("model")["region"].count().reset_index()                 .rename(columns={"region":"count"}).sort_values("count")

models["quintile"] = 1/len(models)*100
models["quintile (cum)"] = models["quintile"].cumsum()

models["share of total"] = (models["count"]/sum(models["count"]))*100
models["share of total (cum)"] = models["share of total"].cumsum()

models.drop(columns=["quintile", "share of total"], inplace=True)

models.head(10)


# In[26]:


plt.plot( models["quintile (cum)"], models["share of total (cum)"])
plt.fill_between(models["quintile (cum)"], models["share of total (cum)"])
#plt.title('Gini: {}'.format(gini), fontsize=20)
plt.ylabel('Cummulative Share', fontsize=15)
plt.xlabel('Model Quintiles', fontsize=15)
plt.show()

#px.line(models, x="quintile (cum)", y="share of total (cum)")


# In[27]:


models[models["count"] ==1].sort_values("share of total (cum)", ascending=False)


# **More than 50% of the total car models have only one entry and account for only 3.7% of the total entries**

# In[28]:


models[models["count"] <10].sort_values("share of total (cum)", ascending=False)


# In[29]:


# Since I will be splitting the data using a 80/20 split, the train dataset should at least have one entry, 
# hence the car model should have at least 5 entries. However in order to be sure, I will drop all models with entries less 
# than 10
drop_models = models[models["count"] <10]["model"]
df1 = df1[~df1["model"].isin(drop_models)]
df1.shape


# ### Handling missing values

# In[30]:


missing_values(df1)


# In[31]:


# Getting the values for the same model from other rows.

cols = ["cylinders", "drive", "type", "manufacturer", "fuel", "transmission"]


for col in cols:
    df11 = df1[df1[col].notna()].copy(deep = True)
    dict_ = {}
    for k, v in df11[['model', col]].values:
        dict_[k] = v
        
    df1[col] = df1.apply( lambda x: x[col] if pd.notna( x[col] )
                                        else dict_[x['model']] if x['model'] in dict_.keys() 
                                        else x[col], axis=1)

del df11, dict_


# In[32]:


missing_values(df1)


# In[33]:


df1.dropna(subset = ["fuel", "type", "title_status", "lat", "long", "manufacturer", "drive", "cylinders"], inplace=True)


# In[34]:


df1.drop_duplicates(keep = "first", inplace = True)


# In[35]:


missing_values(df1)


# In[36]:


models_keep = df1[df1["condition"].notna()]["model"].unique()
df1[~df1["model"].isin(models_keep)].shape[0]/df1.shape[0]*100


# In[37]:


df1 = df1[df1["model"].isin(models_keep)].sort_values(["model", "odometer", "price", "year"])


# In[38]:


df1["condition"].unique()


# In[39]:


df1["condition"].replace(['new', 'like new', 'excellent', 'good', 'fair','salvage'], [1, 2, 3, 4, 5, 6], inplace=True)
df1["condition"] = pd.to_numeric(df1["condition"])


# In[40]:


df1.groupby("condition")["region"].count()


# In[41]:


imputer = KNNImputer(n_neighbors=10)
imputed_cond = []

for model in df1["model"].unique().tolist():
    col = df1[df1["model"] ==model]
    imputed_cond.append(imputer.fit_transform(col[["condition"]]))
    
df1["condition"] = np.concatenate(imputed_cond, axis = 0)
df1["condition"] = round(df1["condition"], 0)  


# In[42]:


df1.groupby("condition")["region"].count()


# In[43]:


df1.duplicated().sum()


# In[44]:


df_clean = df1.drop_duplicates(keep="first")


# In[45]:


missing_values(df_clean)


# In[46]:


df1.shape


# ## Further Cleaning and Visualization

# In[47]:


g = sns.PairGrid(df_clean, height=5)
g.map_diag(sns.boxplot)
g.map_offdiag(sns.scatterplot)


# In[48]:


drop_years = df_clean[(df_clean["year"] <1996) | (df_clean["price"] > 55000) | (df_clean["odometer"] >280000)].index
df_clean.drop(drop_years, axis=0, inplace=True)


# In[49]:


g = sns.PairGrid(df_clean, height=5)
g.map_diag(sns.boxplot)
g.map_offdiag(sns.scatterplot)


# In[50]:


sns.set(rc={'figure.figsize':(13,8)})


# In[51]:


ax = sns.histplot(data=df_clean, x="year", binwidth=4)
ax.set_yscale('log')


# In[52]:


sns.histplot( x=df_clean["price"], binwidth=1000)


# In[53]:


ax = sns.histplot(data=df_clean, x="odometer", binwidth=1000)
ax.set_yscale('log')


# In[54]:


states = geopandas.read_file('./data/tl_2021_us_state.shx')

states.boundary.plot(figsize=(15,8))
plt.title('Cars Locations', fontsize=16)
plt.xlim([-200, -50])
plt.ylim([-20,80])
plt.scatter(x = df_clean['long'], y = df_clean['lat'], marker = "o")


# In[55]:


df_clean[["lat", "long"]].describe([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])


# In[56]:


to_drop = df_clean[(df_clean["lat"] <25.909800) | (df_clean["lat"] > 48.754800) |
          (df_clean["long"] <-149.865983) | (df_clean["long"] > -71.064426)]

df_clean.drop(to_drop.index, inplace = True, axis = 0)


# In[57]:


states.boundary.plot(figsize=(15,8))
plt.title('Cars Locations', fontsize=16)

plt.xlim([-200, -50])
plt.ylim([-20,80])

# plt.xlim([-130, -60])
# plt.ylim([20, 60])
plt.scatter(x = df_clean['long'], y = df_clean['lat'], marker = "o")


# In[58]:


states.boundary.plot(figsize=(15,8))
plt.title('Cars Locations', fontsize=16)

# plt.xlim([-200, -50])
# plt.ylim([-20,80])

plt.xlim([-130, -60])
plt.ylim([20, 60])
plt.scatter(x = df_clean['long'], y = df_clean['lat'], marker = "o")


# In[70]:


df_state = df_clean.groupby(by = ['state'])['region'].count().reset_index().sort_values(by = "region", ascending=False)
df_state.rename(columns={"region":"count"}, inplace=True)
df_state["state"] = df_state["state"].str.upper()

fig = px.choropleth(locations=df_state["state"], locationmode="USA-states", color=df_state["count"],
                    scope="usa", title="Cars Locations (States)")
fig.update_layout(title_x=0.5)
fig.show()


# In[60]:


manufacturers = df_clean.groupby('manufacturer')['region'].count().reset_index().sort_values(by = "region", ascending=False)
manufacturers.rename(columns={"region":"count"}, inplace=True)
sum_man = int(round(manufacturers["count"].head(10).sum()/manufacturers["count"].sum()*100, 0))

fig = px.pie(manufacturers, values='count',  
             title= "<b>Manufacturers Share</b><br>(Top 10 Manufacturers account for about {}%)" .format(sum_man),
             names="manufacturer")
fig.update_traces(textposition='inside')
fig.update_layout(uniformtext_minsize = 15.3, uniformtext_mode='hide')
fig.update_layout(title_x=0.5)
fig.show()


# In[67]:


u1p_sum = manufacturers[manufacturers["count"] < sum(manufacturers["count"])/100]["count"].sum()
u1p_cnt = manufacturers[manufacturers["count"] < sum(manufacturers["count"])/100]["manufacturer"].nunique()
u1p_acc = round(10980/sum(manufacturers["count"]),4)
print("Summary of cars manufacturers with listings shares less than 1%\n\n" +
      tabulate([(u1p_cnt, u1p_sum, u1p_acc*100)], headers= ["Sum", "Count", "Accumulated Share"]))


# In[68]:


# Deleting manufacturers with cars share of less than 1%
to_drop = manufacturers[manufacturers["count"] < sum(manufacturers["count"])/100]["manufacturer"].unique()
df_clean = df_clean[~df_clean["manufacturer"].isin(to_drop)]

# Number of Manfucaturers left 
df_clean["manufacturer"].nunique()


# In[69]:


def plot_types(df):
    col_names = ["condition", "cylinders", "drive", "fuel", "title_status", "transmission", "type"]
    names = ["condition", "title_status", "cylinders", "transmission",  "drive", "type", "fuel"] 

    fig = make_subplots(
        rows = 4, cols = 2,
        subplot_titles=(names),
         specs=[[{"type": "domain"}, {"type": "domain"}],
                [{"type": "domain"}, {"type": "domain"}],
                [{"type": "domain"}, {"type": "domain"}],
                [{"type": "domain"}, {"type": "domain"}]])

    for i in range (len(col_names)):
        df_count = df.groupby(col_names[i])[['region']].count().reset_index().rename(columns={"region":"count"})
        df_count = df_count.sort_values(by = 'count')
        labels = df_count.iloc[:, 0]
        values = df_count.iloc[:, 1]

        if i < 4:
            fig.add_trace(go.Pie(values = values, labels = labels), row=i+1, col=1)
        elif i >= 4:
            fig.add_trace(go.Pie(values = values, labels = labels), row=i-3, col=2)



    fig.update_layout(height=1500, width=1000,
                      title_text="Shares in cars")

    fig.update_layout(title_x=0.5)
    fig.update_traces(textposition='inside')
    fig.show()


# In[70]:


plot_types(df_clean)


# In[71]:


df_clean.groupby("condition").agg({"price":"mean", "odometer":"mean"})


# In[72]:


# Deleting all features with shares less than 1%
df_clean = df_clean[~df_clean["condition"].isin([1, 6])]
df_clean = df_clean[df_clean["cylinders"].isin(["6 cylinders", "4 cylinders", "8 cylinders"])]
df_clean = df_clean[~df_clean["type"].isin(["offroad", "bus"])]
df_clean = df_clean[~df_clean["title_status"].isin(["lien", "missing", "parts only"])]

# Combine electric and hybrid 
df_clean.replace(["electric", "hybrid"], ["electric_or_hybrid", "electric_or_hybrid"], inplace = True)


# In[73]:


plot_types(df_clean)


# In[74]:


models_dist(df_clean).sort_values("count", ascending =False).head(10).sort_values("share of total (cum)").head(5)


# In[75]:


# Intitial Dataframe 
models_plot(models_dist(df_initial))


# In[76]:


print("length of initial dataset:", len(df_initial))
print("Number of models in the initial dataset:", df_initial["model"].nunique()) 


# In[77]:


# Cleaned Dataframe
models_plot(models_dist(df_clean))


# In[78]:


print("length of clean dataset:", len(df_clean))
print("Number of models in the clean dataset:", df_clean["model"].nunique()) 


# ## Preparing Data for modeling

# In[79]:


df_models = df_clean.copy(deep= True)
df_models.head(1)


# In[80]:


drop_col = [0, 1, 4, 13]

X = np.delete(df_models.values, drop_col, 1) 
y = df_models.iloc[:, 1].values

df_dtype = pd.DataFrame(df_models.drop(columns=df_models.iloc[:, (drop_col)].columns).dtypes)            .reset_index().rename(columns={0:"dtype"})

# Getting categorical and numeric columns numbers

cat_col = df_dtype[((df_dtype.dtype !="float32") & (df_dtype.dtype !="float64")) ].index.values
num_col = df_dtype[((df_dtype.dtype =="float32") | (df_dtype.dtype =="float64")) ].index.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


labels = pd.get_dummies(df_models.drop(df_models.columns[drop_col], axis =1)).columns


ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), cat_col),
                                      ('scaler', StandardScaler(), num_col)],
                       sparse_threshold=0,   remainder = 'passthrough')

X_train = ct.fit_transform(X_train)
X_test = ct.transform(X_test)


# ## Different Predicting Methods

# In[84]:


def model_eval(y_actual, y_pred, name):
    
    index = []
    scores = {"MSE": [], "RMSE": [], 
          "MASE": [], "RE" :[], "R2_squared": []} 
    
    index += [name]
    
    scores["MSE"].append(round(metrics.mean_squared_error(y_actual, y_pred),2))
    scores["RMSE"].append(round(math.sqrt(metrics.mean_squared_error(y_actual, y_pred)),2))
    scores["MASE"].append(round(metrics.mean_absolute_error(y_actual, y_pred),2))
    scores["RE"].append(round(metrics.mean_absolute_error(y_actual, y_pred)*len(y_actual)/sum(abs(y_actual)), 4))
    scores["R2_squared"].append(round(metrics.explained_variance_score(y_actual, y_pred),2))

    df_scores = pd.DataFrame(scores, index=index)
                        
    return df_scores


# In[127]:


def cv_result(model, index_name):
    
    scoring_metrics = {"MSE":"neg_mean_squared_error", "RMSE":"neg_root_mean_squared_error",
                       "MASE": "neg_mean_absolute_error", "r2_score" : "r2"}
    result = cross_validate(model, X_train, y_train, scoring=scoring_metrics, cv=5, n_jobs = -1)
    
    index = []
    scores = {"MSE": [], "RMSE": [], "MASE": [], "RE" :[], "R2_squared": []} 
    
    index += [index_name]
    
    scores["MSE"].append(round(result["test_MSE"].mean()*-1 ,2))
    scores["RMSE"].append(round(result["test_RMSE"].mean()*-1 ,2))
    scores["MASE"].append(round(result["test_MASE"].mean()*-1 ,2))
    scores["RE"].append(round(result["test_MASE"].mean()*-1*len(y_train)/sum(abs(y_train)), 4))
    scores["R2_squared"].append(round(result["test_r2_score"].mean() ,2))

    df_scores = pd.DataFrame(scores, index=index)
    return df_scores


# ### Baseline Average Price

# In[126]:


baseline = pd.DataFrame(y_test, columns={"actual"})
baseline["prediction"] = mean(y_train)


baseline = model_eval(baseline["actual"], baseline["prediction"], "Avg Price Model")
baseline


# ### Simple Linear Regression

# In[129]:


LR = LinearRegression()

cv_LR = cv_result(LR, "Linear Regression")
cv_LR


# ### Decision Tree Regressor

# In[132]:


DTR = DecisionTreeRegressor()

cv_DTR = cv_result(DTR, "Decision Tree Regressor")
cv_DTR


# In[135]:


DTR = DecisionTreeRegressor(max_depth=30, min_samples_leaf=2, random_state=0)

cv_DTRt = cv_result(DTR, "Decision Tree Regressor tuned")
cv_DTRt


# ### Random Forest Regressor

# In[136]:


RFR = RandomForestRegressor(random_state=0, n_jobs= -1)

cv_RFR = cv_result(RFR, "Random Forest Regressor")
cv_RFR


# In[137]:


RFRt = RandomForestRegressor(n_estimators=200, max_features=0.25, random_state=0, n_jobs= -1)

cv_RFRt = cv_result(RFRt, "Random Forest Regressor tuned")
cv_RFRt


# ### XGBoost Regressor

# In[138]:


xgb = XGBRegressor(random_state = 0, n_jobs = -1)

cv_XGBR = cv_result(xgb, "XGBoost Regressor")
cv_XGBR


# In[146]:


xgbt = XGBRegressor(n_estimators = 1000, learning_rate= 0.20, max_depth = 9, max_bin = 5000, gamma = 10, reg_lambda = 5,
                   tree_method = "gpu_hist", use_rmm = True,  random_state = 0)

cv_XGBRt = cv_result(xgbt, "XGBoost Regressor tuned")
cv_XGBRt


# ### Summary

# In[140]:


pd.concat([baseline, cv_LR, cv_DTR, cv_DTRt, cv_RFR, cv_RFRt, cv_XGBR, cv_XGBRt])


# ### Best Algorithm in action

# In[140]:


xgb = XGBRegressor(n_estimators = 1000, learning_rate= 0.30, max_depth = 9, max_bin = 5000, gamma = 10, reg_lambda = 5,
                   tree_method = "gpu_hist", use_rmm = True,  random_state = 0)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

xgb_model_tuned = model_eval(y_test, y_pred_xgb, "XGBRegressor tuned")
xgb_model_tuned


# In[142]:


xgb = XGBRegressor(n_estimators = 1000, learning_rate= 0.3, max_depth = 10, max_bin = 5000, gamma = 10, reg_lambda = 5,
                   tree_method = "gpu_hist", use_rmm = True,  random_state = 0)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

model_eval(y_test, y_pred_xgb, "XGBRegressor tuned")


# In[145]:


xgb = XGBRegressor(n_estimators = 1000, learning_rate= 0.2, max_depth = 9, max_bin = 5000, gamma = 10, reg_lambda = 5,
                   tree_method = "gpu_hist", use_rmm = True,  random_state = 0)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

model_eval(y_test, y_pred_xgb, "XGBRegressor tuned")


# ## Outtakes tuning

# In[142]:


# Tuning  n_estimators and learning_rate with gpu_hist tree_method

xgb = XGBRegressor(tree_method = "gpu_hist", random_state = 0) 

n_estimators = [500, 800, 1000]
learning_rate = [0.2, 0.3, 0.4]

param_grid = dict(learning_rate=learning_rate, n_estimators=n_estimators)
kfold = KFold(n_splits=3, shuffle=True)

grid_search = GridSearchCV(xgb, param_grid, scoring="neg_root_mean_squared_error", n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(X_train, y_train)


# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    
# plot results
scores = np.array(means).reshape(len(learning_rate), len(n_estimators))
for i, value in enumerate(learning_rate):
    plt.plot(n_estimators, scores[i], label='learning_rate: ' + str(value))
plt.legend()
plt.xlabel('n_estimators')
plt.ylabel('Log Loss')


# In[144]:


# Tuning max_depth
xgb = XGBRegressor(tree_method = "gpu_hist", n_estimators = 1000, learning_rate = 0.2, random_state = 0) 

param_grid = dict(max_depth=[6, 7, 8, 9, 10])

kfold = KFold(n_splits=3, shuffle=True)

grid_search = GridSearchCV(xgb, param_grid, scoring="neg_root_mean_squared_error", n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(X_train, y_train)


# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[134]:


model = XGBRegressor(tree_method = "gpu_hist", random_state = 0)

n_estimators = [700, 800, 1000]
learning_rate = [0.2, 0.3]
max_bin = [1000, 2000, 3000, 5000]
param_grid = dict(learning_rate=learning_rate, n_estimators=n_estimators, max_bin = max_bin)
kfold = KFold(n_splits=3, shuffle=True)

grid_search = GridSearchCV(model, param_grid, scoring="neg_root_mean_squared_error", n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(X_train, y_train)
                              
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

