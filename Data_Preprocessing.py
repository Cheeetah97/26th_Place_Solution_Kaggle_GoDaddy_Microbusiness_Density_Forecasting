import pandas as pd
import numpy as np

path = "F:/Kaggle/microbusiness_density/February/Data/"
out_path = "F:/Kaggle/microbusiness_density/February/Final_Submission/"

horizon = 6

train_data = pd.read_csv(out_path+f"train_N+{horizon}.csv")
test_data = pd.read_csv(out_path+f"test_N+{horizon}.csv")
coords = pd.read_csv(path+"cfips_location.csv")
co_est = pd.read_csv(path+"co-est2021-alldata.csv",encoding='latin-1')
census_data = pd.read_csv(path+"census_starter.csv")

train_data = train_data.rename(columns={"first_day_of_month":"Date"})
train_data["Date"] = pd.to_datetime(train_data["Date"],format="%Y-%m-%d")
train_data["Year"] = train_data["Date"].dt.year
train_data["Month"] = train_data["Date"].dt.month
train_data["Quarter"] = train_data["Date"].dt.quarter
train_data['istest'] = 0
train_data = train_data.sort_values(by=["cfips","Date"])
train_data = train_data.reset_index(drop=True)

test_data = test_data.rename(columns={"first_day_of_month":"Date"})
test_data["Date"] = pd.to_datetime(test_data["Date"],format="%Y-%m-%d")
test_data["Year"] = test_data["Date"].dt.year
test_data["Month"] = test_data["Date"].dt.month
test_data["Quarter"] = test_data["Date"].dt.quarter
test_data['istest'] = 1
test_data = test_data.sort_values(by=["cfips","Date"])
test_data = test_data.reset_index(drop=True)

co_est["cfips"] = co_est.STATE*1000 + co_est.COUNTY

#%%
# Population Correction As Per 2021

# COLS = ['GEO_ID','NAME','S0101_C01_026E']

# df2021 = pd.read_csv(path+'ACSST5Y2021.S0101-Data.csv',usecols=COLS)
# df2021 = df2021.iloc[1:]
# df2021['S0101_C01_026E'] = df2021['S0101_C01_026E'].astype('int')

# df2020 = pd.read_csv(path+'ACSST5Y2020.S0101-Data.csv',usecols=COLS)
# df2020 = df2020.iloc[1:]
# df2020['S0101_C01_026E'] = df2020['S0101_C01_026E'].astype('int')

# df2019 = pd.read_csv(path+'ACSST5Y2019.S0101-Data.csv',usecols=COLS)
# df2019 = df2019.iloc[1:]
# df2019['S0101_C01_026E'] = df2019['S0101_C01_026E'].astype('int')

# df2018 = pd.read_csv(path+'ACSST5Y2018.S0101-Data.csv',usecols=COLS)
# df2018 = df2018.iloc[1:]
# df2018['S0101_C01_026E'] = df2018['S0101_C01_026E'].astype('int')

# df2017 = pd.read_csv(path+'ACSST5Y2017.S0101-Data.csv',usecols=COLS)
# df2017 = df2017.iloc[1:]
# df2017['S0101_C01_026E'] = df2017['S0101_C01_026E'].astype('int')

# df2021['cfips'] = df2021.GEO_ID.apply(lambda x: int(x.split('US')[-1]) )
# adult2021 = df2021.set_index('cfips').S0101_C01_026E.to_dict()

# df2020['cfips'] = df2020.GEO_ID.apply(lambda x: int(x.split('US')[-1]) )
# adult2020 = df2020.set_index('cfips').S0101_C01_026E.to_dict()

# df2019['cfips'] = df2019.GEO_ID.apply(lambda x: int(x.split('US')[-1]) )
# adult2019 = df2019.set_index('cfips').S0101_C01_026E.to_dict()

# df2018['cfips'] = df2018.GEO_ID.apply(lambda x: int(x.split('US')[-1]) )
# adult2018 = df2018.set_index('cfips').S0101_C01_026E.to_dict()

# df2017['cfips'] = df2017.GEO_ID.apply(lambda x: int(x.split('US')[-1]) )
# adult2017 = df2017.set_index('cfips').S0101_C01_026E.to_dict()

# train_data['adult2021'] = train_data.cfips.map(adult2021)
# train_data['adult2020'] = train_data.cfips.map(adult2020)
# train_data['adult2019'] = train_data.cfips.map(adult2019)
# train_data['adult2018'] = train_data.cfips.map(adult2018)
# train_data['adult2017'] = train_data.cfips.map(adult2017)

# train_data.loc[train_data.Year==2022,"microbusiness_density"] = train_data.loc[train_data.Year==2022,"microbusiness_density"] * train_data.loc[train_data.Year==2022,"adult2020"] / train_data.loc[train_data.Year==2022,"adult2021"]
# train_data.loc[train_data.Year==2021,"microbusiness_density"] = train_data.loc[train_data.Year==2021,"microbusiness_density"] * train_data.loc[train_data.Year==2021,"adult2019"] / train_data.loc[train_data.Year==2021,"adult2021"]
# train_data.loc[train_data.Year==2020,"microbusiness_density"] = train_data.loc[train_data.Year==2020,"microbusiness_density"] * train_data.loc[train_data.Year==2020,"adult2018"] / train_data.loc[train_data.Year==2020,"adult2021"]
# train_data.loc[train_data.Year==2019,"microbusiness_density"] = train_data.loc[train_data.Year==2019,"microbusiness_density"] * train_data.loc[train_data.Year==2019,"adult2017"] / train_data.loc[train_data.Year==2019,"adult2021"]

# train_data = train_data.drop(["adult2021","adult2020","adult2019","adult2018","adult2017"],axis=1)

#%%
# Target Transformation

df = pd.DataFrame()
for cnty in list(train_data.cfips.unique()):
    temp = train_data[train_data.cfips==cnty]
    temp = temp.sort_values(by=["Date"])
    temp = temp.reset_index(drop=True)
    
    temp['Target'] = temp["microbusiness_density"].copy()
    temp['Target2'] = temp["microbusiness_density"].copy()
    temp['mbd_orig'] = temp["microbusiness_density"].copy()
    
    for i in range(len(temp)-4,2,-1):
        thr = 0.10*np.mean(temp["Target"].iloc[:i])
        difa = temp["Target"].iloc[i]-temp["Target"].iloc[i-1]
        if (difa >= thr) or (difa <= -thr):  
            if difa > 0:
                temp["Target"].iloc[:i] += difa - 0.003 
            else:
                temp["Target"].iloc[:i] += difa + 0.003
                
    for i in range(len(temp)-1,2,-1):
        thr = 0.10*np.mean(temp["Target2"].iloc[:i])
        difa = temp["Target2"].iloc[i]-temp["Target2"].iloc[i-1]
        if (difa >= thr) or (difa <= -thr):  
            if difa > 0:
                temp["Target2"].iloc[:i] += difa - 0.003 
            else:
                temp["Target2"].iloc[:i] += difa + 0.003
            
    temp["Target"].iloc[0] = temp["Target"].iloc[1]*0.99
    temp["microbusiness_density"] = temp["Target"]
    temp["Target"] = (temp["Target"].shift(-1)/temp["Target"]) - 1
    
    temp["Target2"].iloc[0] = temp["Target2"].iloc[1]*0.99
    temp["microbusiness_density2"] = temp["Target2"]
    temp["Target2"] = (temp["Target2"].shift(-1)/temp["Target2"]) - 1
    
    if (cnty == 28055)|(cnty == 48269)|(cnty == 48301):
        temp["Target"] = 0.0
        temp["Target2"] = 0.0
        
    df = pd.concat([df,temp])
        
df = df.reset_index(drop=True)
train_data = df.copy()  
#%%
# Statistical Features

combined_data = pd.concat([train_data,test_data[["row_id","Date","Year","Month","Quarter","cfips","istest"]]])
combined_data = combined_data.sort_values(by=["cfips","Date"])
combined_data = combined_data.reset_index(drop=True)

combined_data["state"] = combined_data["state"].ffill()
combined_data["county"] = combined_data["county"].ffill()
combined_data['county_i'] = (combined_data['county'] + combined_data['state']).factorize()[0]
combined_data['state_i'] = combined_data['state'].factorize()[0]
combined_data["dcount"] = combined_data.groupby(['cfips'])['row_id'].cumcount()
combined_data['scale'] = (combined_data['Date'] - combined_data['Date'].min()).dt.days
combined_data['scale'] = combined_data['scale'].factorize()[0]
combined_data['lastactive'] = combined_data.groupby('cfips')['active'].transform('last')

def build_features(df,lags):  

    for lag in range(1, lags):
        df[f'mbd_lag_{lag}'] = df.groupby('cfips')['Target'].shift(lag)
        df[f'act_lag_{lag}'] = df.groupby('cfips')['active'].diff(lag)
        
    lag = 1
    for window in [2,4,6,8,10]:
        df[f'mbd_rollmea{window}_{lag}'] = df.groupby('cfips')[f'mbd_lag_{lag}'].transform(lambda s: s.rolling(window, min_periods=1).sum())
    
    return df

combined_data = build_features(combined_data,9)

#%%
# Latitude and Longitude Coordinates an Other Features

combined_data = combined_data.merge(coords.drop("name",axis=1),on="cfips")
combined_data = combined_data.merge(co_est,on="cfips",how="left")

coordinates = combined_data[['lng','lat']].values
emb_size = 20
precision = 1e6

latlon = np.expand_dims(coordinates, axis=-1)

m = np.exp(np.log(precision)/emb_size)
angle_freq = m ** np.arange(emb_size)
angle_freq = angle_freq.reshape(1,1, emb_size)
latlon = latlon * angle_freq
latlon[..., 0::2] = np.cos(latlon[..., 0::2])

def rot(df):
    for angle in [15,30,45]:
        df[f'rot_{angle}_x'] = (np.cos(np.radians(angle)) * df['lat']) + (np.sin(np.radians(angle)) * df['lng'])
        df[f'rot_{angle}_y'] = (np.cos(np.radians(angle)) * df['lat']) - (np.sin(np.radians(angle)) * df['lng'])
    return df

combined_data = rot(combined_data)

#%%
# Census Data

for col in list(census_data.columns):
    if len(census_data.loc[census_data[col].isnull()])>0:  
        census_data.loc[census_data[col].isnull(),col] = census_data.loc[census_data[col].isnull(),col.replace(col[-4:],str(int(col[-4:])-1))].iloc[0]

combined_data = combined_data.merge(census_data,how="left",on=["cfips"])
combined_data.to_csv(out_path+f"All_Features_Final_{horizon}.csv",index=False)

