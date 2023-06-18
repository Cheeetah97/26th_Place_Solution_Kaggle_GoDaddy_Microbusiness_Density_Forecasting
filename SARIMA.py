import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
#%%
# Functions
def get_forecast(df,lag,o_0=1,o_1=1,o_2=1,s_0=1,s_1=1,s_2=1,plot="no"):
    forecasts = []
    actuals = []
    for i in range(len(df)-lag,len(df)+1,1):
        try:
            actual = df["microbusiness_density"].iloc[i]
        except:
            actual = np.nan
       
        order = (o_0,o_1,o_2)
        seasonal_order = (s_0,s_1,s_2,24)
        
        try:
            model = sm.tsa.statespace.SARIMAX(df["microbusiness_density"].iloc[:i],order=order,seasonal_order=seasonal_order)
            results = model.fit()
        except:
            break
        
        forecasts.append(results.forecast(steps=1).iloc[0])
        actuals.append(actual)
        
    if len(actuals)<14:
        return actuals,forecasts,np.nan
    else:
        if plot == "yes":
            plt.plot(actuals)
            plt.plot(forecasts)
            plt.legend()
            plt.show()
        return actuals,forecasts,smape(np.array(actuals[:-1]),np.array(forecasts[:-1]))

def rmse(targets,predictions):
    return np.sqrt(((targets - predictions)**2).mean())

def smape(y_true, y_pred):
    smap = np.zeros(len(y_true))
    
    num = np.abs(y_true - y_pred)
    dem = ((np.abs(y_true) + np.abs(y_pred)) / 2)
    
    pos_ind = (y_true!=0)|(y_pred!=0)
    smap[pos_ind] = num[pos_ind] / dem[pos_ind]
    
    return 100 * np.mean(smap)

#%%
path = "F:/Kaggle/microbusiness_density/February/Data/"

all_feats = pd.read_csv(path+"All_Features_New.csv")
all_feats = all_feats.drop(["Target","microbusiness_density"],axis=1)
all_feats = all_feats.rename(columns={"Target2":"Target","microbusiness_density2":"microbusiness_density"})
season = pd.read_csv(path+"ExtractedFeatures.csv")
season = season[["cfips","season"]]

#last_pop_df = all_feats.loc[(all_feats.Year==2022)&(all_feats.Month==12),["cfips","active","microbusiness_density"]]
#last_pop_df["Population"] = (last_pop_df["active"]/last_pop_df["microbusiness_density"])*100
#high_population_counties = list(last_pop_df.loc[last_pop_df.Population>25000,"cfips"].unique())
all_feats = all_feats.merge(season,how="left",on=["cfips"])
all_feats = all_feats.loc[all_feats.season>0.6]
all_feats = all_feats.drop("microbusiness_density",axis=1)
all_feats = all_feats.rename(columns={"mbd_orig":"microbusiness_density"})

global args
args = {"o_0":1,
        "o_1":1,
        "o_2":1,
        "s_0":1,
        "s_2":1,
        "s_2":1,
        }

res_all = pd.DataFrame()
for cnty in list(all_feats.cfips.unique()):
    temp = all_feats.loc[all_feats.cfips==cnty]
    temp = temp.loc[temp.istest==0]
    temp = temp.reset_index(drop=True)
    temp = temp.sort_values(by=["Year","Month"])
    #%%
    # order = (1,0,0)
    # seasonal_order = (1,0,0,12)
    
    # forecasts = []
    # actuals = []
    # for i in range(len(temp)-14,len(temp)+1,1):
    #     try:
    #         actual = temp["microbusiness_density"].iloc[i+1]
    #         #curr = temp["microbusiness_density"].iloc[i]
    #     except:
    #         actual = np.nan
    #         #curr = temp["microbusiness_density"].iloc[i]
        
    #     try:
    #         model = sm.tsa.statespace.SARIMAX(temp["microbusiness_density"].iloc[:i],order=order,seasonal_order=seasonal_order)
    #         results = model.fit()
    #     except:
    #         break
        
    #     #forecasts.append((results.forecast(steps=1).iloc[0]+1)*curr)
    #     forecasts.append(results.forecast(steps=1).iloc[0])
    #     actuals.append(actual)
    # plt.plot(actuals)
    # plt.plot(forecasts)
    # plt.legend()
    # plt.show()
    #%%
    
    
    all_scores = pd.DataFrame()
    
    for var in ["o_0","o_1","o_2","s_0","s_1","s_2"]:
        for val in range(0,4):
            args[var] = val
            actuals,forecasts,score = get_forecast(temp,14,**args)
            scores = pd.DataFrame(data={"Score":[score]})
            scores[var] = val
            all_scores = pd.concat([all_scores,scores])
        t_scores = all_scores.dropna(subset=[var]).reset_index(drop=True)
        args[var] = t_scores.loc[t_scores.Score==t_scores.Score.min(),var].iloc[0]
            
    all_scores = all_scores.dropna(subset="Score")       
    all_scores = all_scores.reset_index(drop=True)
    
    o_0 = all_scores.dropna(subset=["o_0"])
    o_0 = o_0.loc[o_0.Score==o_0.Score.min(),"o_0"].iloc[0]
    
    o_1 = all_scores.dropna(subset=["o_1"])
    o_1 = o_1.loc[o_1.Score==o_1.Score.min(),"o_1"].iloc[0]
    
    o_2 = all_scores.dropna(subset=["o_2"])
    o_2 = o_2.loc[o_2.Score==o_2.Score.min(),"o_2"].iloc[0]
    
    s_0 = all_scores.dropna(subset=["s_0"])
    s_0 = s_0.loc[s_0.Score==s_0.Score.min(),"s_0"].iloc[0]
    
    s_1 = all_scores.dropna(subset=["s_1"])
    s_1 = s_1.loc[s_1.Score==s_1.Score.min(),"s_1"].iloc[0]
    
    s_2 = all_scores.dropna(subset=["s_2"])
    s_2 = s_2.loc[s_2.Score==s_2.Score.min(),"s_2"].iloc[0]
    
    actuals,forecasts,score = get_forecast(temp,14,o_0=o_0,o_1=o_1,o_2=o_2,s_0=s_0,s_1=s_1,s_2=s_2,plot="no")
    
    if len(actuals)<14:
        continue
        #raise ValueError("CHECK!")
    else:
        order = (o_0,o_1,o_2)
        seasonal_order = (s_0,s_1,s_2,24)
        try:
            model = sm.tsa.statespace.SARIMAX(np.array(actuals[:-1])-np.array(forecasts[:-1]),order=order,seasonal_order=seasonal_order)
            results2 = model.fit()
            orig = forecasts.copy()
            forecasts[-1] = forecasts[-1] + results2.forecast(steps=1)[0]
            res = pd.DataFrame(data={"Year":list(temp.Year.tail(14))+[2023],"Month":list(temp.Month.tail(14))+[1],"cfips":list(temp.cfips.tail(15)),"Actual":actuals,"Orig_Predicted":orig,"Predicted":forecasts})
        except:
            res = pd.DataFrame(data={"Year":list(temp.Year.tail(14))+[2023],"Month":list(temp.Month.tail(14))+[1],"cfips":list(temp.cfips.tail(15)),"Actual":actuals,"Orig_Predicted":forecasts,"Predicted":forecasts})
        res_all = pd.concat([res_all,res])
    
res_all.to_csv(path+"Arima_Res.csv",index=False)
    
    
    # plt.plot(res["Actual"])
    # plt.plot(res["Orig_Predicted"])
    # plt.plot(res["Predicted"])
    # plt.title(cnty)
    # plt.legend()
    # plt.show()
    