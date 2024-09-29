import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder
import torch 
from NCF_model.NCF import *
from sklearn.metrics.pairwise import cosine_similarity
import re
from fastapi import FastAPI
from typing import Union
from typing import Annotated
import uvicorn
from ttest import *
from fastapi import FastAPI, Path
app = FastAPI()




ratings_df = pd.read_csv("cleaned_100k/ratings.csv")
Movies_df=  pd.read_csv("cleaned_100k/descriptions_df.csv")
Movies_df.drop(columns=['Unnamed: 0'],inplace=True)
title = pd.read_csv("cleaned_100k/movies.csv")
content_df  =  pd.merge(ratings_df,Movies_df,on='MovieID')
content_df.drop(columns=['Timestamp'],inplace=True)

y_group_user =  content_df.groupby('UserID')
y_target_content = []
for i,j  in  y_group_user:
    y_target_content.append(j['Rating'].values)
    
x_group_user =  content_df.groupby('UserID')
x_train_content = []
for i,j  in  x_group_user:
    x_train_content.append(j.drop(columns='UserID'))
     
n_users,n_movies,n_factors =610,9742,50
model = NCF(n_users,n_movies,n_factors)
model.load_state_dict(torch.load('NCF_model/model.pth', weights_only=True,map_location =torch.device('cpu') ))
model.eval()



@app.get("/")
def read_root():
    return {"Hello, Welcome to Movie Recommendation."}
@app.get("/ttest")
def Hypothesis_testing():
    return ttest()
def Hybrid_Recommender_v1(user_id):
    watched = set(x_train_content[user_id]['MovieID'].values)
    not_watched = Movies_df[Movies_df['MovieID'].map(lambda x: x not in watched)]
    result = pd.DataFrame({'Movie_id':not_watched['MovieID']})
    CBF(user_id,not_watched,result)
    NCF(user_id,not_watched,result)
    result['hybrid_preds'] =  result['collab_preds'] * 0.5 + result['content_preds'] *0.5 
    
    return result
def Hybrid_Recommender_v2(user_id):
    watched = set(x_train_content[user_id]['MovieID'].values)
    not_watched = Movies_df[Movies_df['MovieID'].map(lambda x: x not in watched)]
    result = pd.DataFrame({'Movie_id':not_watched['MovieID']})
    CBF(user_id,not_watched,result)
    NCF(user_id,not_watched,result)
    result['hybrid_preds'] =  result['collab_preds'] * 0.6 + result['content_preds'] *0.4
    
    return result

def  CBF(user_id,not_watched,result):
    #train  =  [ast.literal_eval(x) for x  in   x_train_content[user_id]['TFIDF']]
    train = x_train_content[user_id].iloc[:,2:]
    model =  SVR(C=1.3)
    model.fit(train,y_target_content[user_id])
  
    not_watched_tfidf = not_watched.iloc[:,:-1]
    preds = model.predict(not_watched_tfidf)
    preds =  np.clip(preds,0,5)
    result['content_preds'] = preds
    
def NCF(user_id,not_watched,result):
    movies = LabelEncoder()
    movies.classes_ = np.load('movies.npy')
    users = LabelEncoder()
    users.classes_ = np.load('users.npy')
    x_predict_collab = torch.stack((torch.tensor(users.transform([4]* len(not_watched))),torch.tensor(movies.transform(not_watched['MovieID']))),1)
    model.eval()
    preds = model(x_predict_collab)
    result['collab_preds'] = preds.tolist()

@app.get("/Top10v1/{user_id}")
async def Top_10_Recommendations_1(user_id: Annotated[int, Path(title="The ID of the item to get", gt=0, le=610)]):
    result = Hybrid_Recommender_v1(user_id)
    movie_ids = result.sort_values('hybrid_preds', ascending=False)['Movie_id'][:10]
    movies = list(title[title['MovieID'].isin(movie_ids)]['Title'].values)
    return {'Top 10 Recommendations':movies}
@app.get("/Top10v2/{user_id}")
async def Top_10_Recommendations_1(user_id: Annotated[int, Path(title="The ID of the item to get", gt=0, le=610)]):
    result = Hybrid_Recommender_v2(user_id)
    movie_ids = result.sort_values('hybrid_preds', ascending=False)['Movie_id'][:10]
    movies = list(title[title['MovieID'].isin(movie_ids)]['Title'].values)
    return {'Top 10 Recommendations':movies}

@app.get("/new_user/{movie_name}")
async def Similar_movies(name: str):
    cosine_sim = cosine_similarity(Movies_df.iloc[:,:-1],Movies_df.iloc[:,:-1])
    idxs = title[title['Title'] == name].index[0]
    sim_scores = list(enumerate(cosine_sim[idxs]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    similar = [x  for x,y in sim_scores]
    return {'Top 10 Recommendations':list(title.iloc[similar]['Title'])}

@app.get("/Top10v1/{user_id}/rate_recommender/{rating}")
async def Rating_Recommendations_1(user_id: Annotated[int, Path(title="The ID of the item to get", gt=0, le=610)],rating: Annotated[int, Path(title="The ID of the item to get", gt=0, le=5)]):
    Recommender_Response1 = pd.read_csv('Recommender_Responsev1.csv',usecols=['Users','Recommender_Ratings'])   
    new_val = {'Users':user_id,'Recommender_Ratings':rating}
    Recommender_Response1 = pd.concat([Recommender_Response1, pd.DataFrame([new_val])], ignore_index=True)
    Recommender_Response1.to_csv('Recommender_Responsev1.csv') 
                                                         
    print(user_id)
    return {"Thank you for rating our recommendation System.":user_id}
    

@app.get("/Top10v2/{user_id}/rate_recommender/{rating}")
async def Rating_Recommendations_2(user_id:  Annotated[int, Path(title="The ID of the item to get", gt=0, le=610)], rating: Annotated[int, Path(title="The ID of the item to get", gt=0, le=5)]):
        Recommender_Response2 = pd.read_csv('Recommender_Responsev2.csv',usecols=['Users','Recommender_Ratings']) 
        new_val ={'Users':user_id,'Recommender_Ratings':rating}
        Recommender_Response2 = pd.concat([Recommender_Response2, pd.DataFrame([new_val])], ignore_index=True)
        Recommender_Response2.to_csv('Recommender_Responsev2.csv') 
        print(user_id)
        return  {"Thank you for rating our recommendation System.":user_id}
    
    
if __name__ == '__main__':
        uvicorn.run(app,port= 8000,host="0.0.0.0")