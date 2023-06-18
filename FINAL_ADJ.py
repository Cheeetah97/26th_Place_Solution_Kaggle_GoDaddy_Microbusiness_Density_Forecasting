import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

out_path = "F:/Kaggle/microbusiness_density/February/Final_Submission/"
path = "F:/Kaggle/microbusiness_density/February/Data/"
#%%

to_change = [1125,  4009,  5033,  6073,  6103,  8087, 10001, 12003, 12007,
             12035, 12093, 12101, 12113, 13129, 13185, 13285, 13305, 16039,
             17109, 17119, 17157, 17197, 18001, 18027, 18157, 18167, 20057,
             20091, 21089, 22005, 22009, 22031, 23017, 23023, 24021, 24023,
             26031, 26045, 26069, 27115, 28047, 28085, 28087, 28133, 29001,
             29127, 31001, 32007, 33019, 36031, 36035, 36085, 37097, 37119,
             37155, 37157, 39001, 39033, 39101, 40037, 40081, 40109, 40143,
             41017, 42001, 42003, 42035, 45029, 45055, 46029, 46083, 46103,
             47053, 47055, 47063, 47065, 47073, 47187, 48025, 48067, 48091,
             48121, 48141, 48157, 48199, 48203, 48363, 48367, 48471, 48497,
             50017, 50023, 50025, 51033, 53027, 54035, 54069, 55019, 55029,
             55081]


all_feats = pd.read_csv(out_path+"All_Features_Final_1.csv")
last_pop_df = all_feats.loc[(all_feats.Year==2022)&(all_feats.Month==12),["cfips","active","microbusiness_density"]]
last_pop_df["Population"] = (last_pop_df["active"]/last_pop_df["microbusiness_density"])*100
high_population_counties = list(last_pop_df.loc[last_pop_df.Population>20000,"cfips"].unique())
high_last_active_counties = list(last_pop_df.loc[last_pop_df.active>150,"cfips"].unique())
all_feats = all_feats.loc[all_feats.cfips.isin(high_population_counties)&(all_feats.cfips.isin(high_last_active_counties))]

column_names = ['GEO_ID','NAME','S0101_C01_026E']
df2021 = pd.read_csv(path+'ACSST5Y2021.S0101-Data.csv',usecols=column_names)
df2021 = df2021.iloc[1:]
df2021['S0101_C01_026E'] = df2021['S0101_C01_026E'].astype('int')
df2021['cfips'] = df2021.GEO_ID.apply(lambda x: int(x.split('US')[-1]))
adult2021 = df2021.set_index('cfips').S0101_C01_026E.to_dict()

all_feats['adult2021'] = all_feats['cfips'].map(adult2021)

pred = pd.read_csv(out_path+"CHECK.csv")
pred['adult2021'] = pred['cfips'].map(adult2021)
pred["first_day_of_month"] = pd.to_datetime(pred["first_day_of_month"],format="%Y-%m-%d")
pred["Year"] = pred["first_day_of_month"].dt.year
pred["Month"] = pred["first_day_of_month"].dt.month
pred["microbusiness_density"] = pred["microbusiness_density_new"].copy()

sub = pd.read_csv(out_path+"low_submission.csv")
sub["cfips"] = sub["row_id"].apply(lambda x:x.split("_")[0])
sub["cfips"] = sub["cfips"].astype('int64')
sub['adult2021'] = sub['cfips'].map(adult2021)

sub["Date"] = sub["row_id"].apply(lambda x:x.split("_")[1])
sub["Date"] = pd.to_datetime(sub["Date"],format="%Y-%m-%d")
sub["Year"] = sub["Date"].dt.year
sub["Month"] = sub["Date"].dt.month

sub = sub.sort_values(by=["cfips","row_id"])
sub = sub.reset_index(drop=True)

#%%

count_df = pd.DataFrame()
df = pd.DataFrame()
for cnty in to_change:
    temp_p = pred[pred.cfips==cnty]
    temp_p = temp_p[["Year","Month","cfips","microbusiness_density","adult2021","Portion"]]
    temp_p = temp_p.loc[temp_p.Portion=="future"]
    temp_p["ypred"] = temp_p["microbusiness_density"].copy()
    temp_p["microbusiness_density"] = np.nan
    temp_p = temp_p.reset_index(drop=True)
    
    
    temp_a = all_feats[all_feats.cfips==cnty]
    temp_a = temp_a[["Year","Month","cfips","microbusiness_density","adult2021"]]
    temp_a = temp_a.sort_values(by=["Year","Month"])
    temp_a = temp_a.dropna()
    temp_a = temp_a.reset_index(drop=True)
    temp_a["Portion"] =  "test"
    temp_a["predicted"] = temp_a["microbusiness_density"].copy()
    temp_a["predicted_adj"] = temp_a["microbusiness_density"].copy()
    
    for mnth in range(1,7):
        
        if mnth == 1:
            temp_f = temp_p.loc[temp_p.Month<=mnth]
        else:
            temp_f = temp_p.loc[temp_p.Month<=mnth]
            temp_f.loc[~(temp_f.Month==temp_f.Month.max()),"microbusiness_density"] = temp_f.loc[~(temp_f.Month==temp_f.Month.max()),"ypred"]
            temp_f.loc[~(temp_f.Month==temp_f.Month.max()),"ypred"] = np.nan
            temp_f.loc[~(temp_f.Month==temp_f.Month.max()),"Portion"] = "test"
    
        #%%
        # Trend Calculation
        
        temp = all_feats[all_feats.cfips==cnty]
        temp = temp[["Year","Month","cfips","microbusiness_density","adult2021"]]
        temp = temp.sort_values(by=["Year","Month"])
        temp = temp.dropna()
        temp = temp.reset_index(drop=True)
        temp["Portion"] =  "test"
        
        temp_all = pd.concat([temp,temp_f]).reset_index(drop=True)
        
        temp["Rolling_Median"] = temp['microbusiness_density'].rolling(3,min_periods=1).median()
        
        model = LinearRegression()
        X = [i for i in range(0,len(temp["Rolling_Median"].tail(4)),1)]
        X = np.reshape(X,(len(X),1))
        Y = temp["Rolling_Median"].tail(4).tolist()
        Y = np.reshape(Y,(len(Y),1))
        model.fit(X,Y)
        temp_all["Trend"] = model.coef_[0][0]/np.median(Y)
        
        #%%
        # First Two Values Trend
        
        model = LinearRegression()
        temp2 = temp_all[-4:-2]
        X = [i for i in range(1,3,1)]
        X = np.reshape(X,(len(X),1))
        Y = temp2["microbusiness_density"].values
        model.fit(X,Y)
        temp_all["First_Trend"] = model.coef_[0]
        
        #%%
        # Last Two Values Trend
        
        model = LinearRegression()
        temp2 = temp_all[-3:-1]
        X = [i for i in range(1,3,1)]
        X = np.reshape(X,(len(X),1))
        Y = temp2["microbusiness_density"].values
        model.fit(X,Y)
        temp_all["Last_Trend"] = model.coef_[0]
        
        temp_all.loc[temp_all.Portion=="test","ypred"] = temp_all.loc[temp_all.Portion=="test","microbusiness_density"].copy()
        temp_all["Adj_Pred"] = temp_all["ypred"].copy()
        
        if temp_all["Trend"].iloc[-1] > 0:
        
            if np.abs(temp_all["Last_Trend"].iloc[-1]>temp_all["First_Trend"].iloc[-1]):
                if (np.abs(temp_all["Last_Trend"].iloc[-1]/temp_all["First_Trend"].iloc[-1]) > 0.75) & (np.abs(temp_all["Last_Trend"].iloc[-1]/temp_all["First_Trend"].iloc[-1]) < 1.45):
                    ratio = np.abs(temp_all["Last_Trend"].iloc[-1]/temp_all["First_Trend"].iloc[-1])/2
                else:
                    ratio = np.abs(1-np.abs(temp_all["Last_Trend"].iloc[-1]/temp_all["First_Trend"].iloc[-1]))/2
                if ratio > 1:
                    temp_all["Adj_Pred"].iloc[-1] = temp_all["ypred"].iloc[-1]*((np.abs(((1)))*temp_all["Trend"].iloc[-1])+1) 
                else:
                    temp_all["Adj_Pred"].iloc[-1] = temp_all["ypred"].iloc[-1]*((np.abs(((ratio)))*temp_all["Trend"].iloc[-1])+1) 
            
            else:
                if (np.abs(temp_all["First_Trend"].iloc[-1]/temp_all["Last_Trend"].iloc[-1]) > 0.75) & (np.abs(temp_all["First_Trend"].iloc[-1]/temp_all["Last_Trend"].iloc[-1]) < 1.45):
                    ratio = np.abs(temp_all["Last_Trend"].iloc[-1]/temp_all["First_Trend"].iloc[-1])/2
                else:
                    ratio = np.abs(1-np.abs(temp_all["Last_Trend"].iloc[-1]/temp_all["First_Trend"].iloc[-1]))*2
                if ratio > 1:
                    if ratio > 1.05:
                        temp_all["Adj_Pred"].iloc[-1] = temp_all["ypred"].iloc[-1]
                    else:
                        temp_all["Adj_Pred"].iloc[-1] = temp_all["ypred"].iloc[-1]*((np.abs(((ratio/2)))*temp_all["Trend"].iloc[-1])+1) 
                else:
                    temp_all["Adj_Pred"].iloc[-1] = temp_all["ypred"].iloc[-1]*((np.abs(((1-ratio)))*temp_all["Trend"].iloc[-1])+1) 
            
            temp_all["Adj_Pred"].iloc[-1] = np.round(temp_all["Adj_Pred"].iloc[-1] * temp_all["adult2021"].iloc[-1] / 100) / temp_all["adult2021"].iloc[-1]* 100
            temp_all["Adj_Pred"].iloc[-1] = (temp_all["Adj_Pred"].iloc[-1]*0.6) + (temp_all["ypred"].iloc[-1]*0.4)
            temp_all["Adj_Pred"].iloc[-1] = np.round(temp_all["Adj_Pred"].iloc[-1] * temp_all["adult2021"].iloc[-1] / 100) / temp_all["adult2021"].iloc[-1]* 100
           
            c = temp_all.loc[temp_all.Portion=="future"]
            c = c.drop(["First_Trend","Last_Trend","Trend"],axis=1)
            c = c.rename(columns={"ypred":"predicted","Adj_Pred":"predicted_adj"})
            temp_a = pd.concat([temp_a,c])

    plt.figure(figsize=(20,7))
    plt.plot(temp_a["predicted"].iloc[-12:],'-o',color='blue',label='Predicted')
    plt.plot(temp_a["predicted_adj"].iloc[-12:],'-o',color='green',label="Adjusted")
    plt.legend()
    plt.title(cnty)
    plt.show()
    df = pd.concat([df,temp_a.loc[temp_a.Year==2023,["Year","Month","cfips","predicted_adj"]]])
    
    
df = df.reset_index(drop=True)
    
for k,v in sub.iterrows():
    try:
        sub["microbusiness_density"].iloc[k] = df.loc[(df.cfips==sub["cfips"].iloc[k])&
                                                      (df.Year==sub["Year"].iloc[k])&
                                                      (df.Month==sub["Month"].iloc[k]),"predicted_adj"].iloc[0]
        sub["microbusiness_density"].iloc[k] = np.round(sub["microbusiness_density"].iloc[k] * sub["adult2021"].iloc[k]/ 100) / sub["adult2021"].iloc[k] * 100
    except:
        continue


sub[["row_id","microbusiness_density"]].to_csv(out_path+"low_submissionadj.csv",index=False) 
    