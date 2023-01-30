import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df=pd.read_csv("FutbolMatches.csv")
df=df.iloc[:,1:]


def train_test_transform(i,x,y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=i)
    return X_train, X_test, y_train, y_test


def pre(df):
    teams = np.union1d(df["strong_team"].unique(),df["weak_team"].unique())
    tot_mat={}
    for team in teams:
        tot_mat[team]=0

    for team in teams:
        tot_mat[team]=len(df[(df["strong_team"]==team)|(df["weak_team"]==team)])
        
    for key in tot_mat.keys():
        if(tot_mat[key]<20):
            teams = teams[teams != key]
            
    df=df[(df["weak_team"].isin(teams))&(df["strong_team"].isin(teams))]
    
    df["strong_form"] = 0
    df["weak_form"] = 0
    df.reset_index(drop=True, inplace=True)
    
    df["strong_win"] = 0
    df["weak_win"] = 0
    for i in range(1082):
        if(df.iloc[i,:]["strong_score"] > df.iloc[i,:]["weak_score"]):
            df["strong_win"][i] = 1
            df["weak_win"][i] = -1
        elif(df.iloc[i,:]["strong_score"] < df.iloc[i,:]["weak_score"]):
            df["strong_win"][i] = -1
            df["weak_win"][i] = 1
    
    newdf = pd.DataFrame(index=range(0,100),columns=teams)
    for team in teams:
        for i in range(len(df[(df["strong_team"]==team)|(df["weak_team"]==team)])):
            if(df[(df["strong_team"]==team)|(df["weak_team"]==team)].iloc[i,1]==team):
                newdf[team][i] = df[(df["strong_team"]==team)|(df["weak_team"]==team)].iloc[i,:]["strong_win"]
            else:
                newdf[team][i] = df[(df["strong_team"]==team)|(df["weak_team"]==team)].iloc[i,:]["weak_win"]
                
    team_form = pd.DataFrame(index=range(0,94),columns=teams)
    x = 0
    pt = 0
    for team in teams:
        i = 1
        team_form[team][0] = newdf[team][0]
        while (newdf[team][i]!= np.nan)&(i<94):
            x = 0
            pt = 0
            if(i<=4):
                for r in range(i,-1,-1):
                    x=x+1
                    pt+=newdf[team][r]

            else:
                for r in range(i,i-5,-1):
                    x=x+1
                    pt+=newdf[team][r]


            team_form[team][i] = pt/x
            i+=1
    
    mat_count = {}
    for team in teams:
        mat_count[team] = 0

    df["strong_form"]=0
    df["weak_form"]=0
    df["strong_form"]=df["strong_form"].astype(float)
    df["weak_form"]=df["weak_form"].astype(float)
    
    for i in range(len(df)):
        df["strong_form"][i] = team_form[df["strong_team"][i]][mat_count[df["strong_team"][i]]]
        mat_count[df["strong_team"][i]]+=1
        df["weak_form"][i] = team_form[df["weak_team"][i]][mat_count[df["weak_team"][i]]]
        mat_count[df["weak_team"][i]]+=1
        
    del df["date"]
    del df["weak_win"]
    del df["strong_win"]
    del df['strong_cc_positioning']
    del df['weak_cc_positioning']
    del df['strong_bu_positioning']
    del df['weak_bu_positioning']
    
    return df.iloc[:,5:]