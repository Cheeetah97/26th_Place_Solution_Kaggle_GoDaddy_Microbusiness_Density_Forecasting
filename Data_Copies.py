import pandas as pd
import numpy as np

path = "F:/Kaggle/microbusiness_density/February/Data/"
out_path = "F:/Kaggle/microbusiness_density/February/Final_Submission/"

train_data = pd.read_csv(path+"train.csv")
test_data = pd.read_csv(path+"test.csv")
rev_test = pd.read_csv(path+"revealed_test.csv")

#%%
# N+1

train_data = pd.concat([train_data,rev_test])
test_data = test_data.loc[~(test_data.row_id.isin(list(train_data.row_id.unique())))]

# Population
COLS = ['GEO_ID','NAME','S0101_C01_026E']

df2021 = pd.read_csv(path+'ACSST5Y2021.S0101-Data.csv',usecols=COLS)
df2021 = df2021.iloc[1:]
df2021['S0101_C01_026E'] = df2021['S0101_C01_026E'].astype('int')

df2020 = pd.read_csv(path+'ACSST5Y2020.S0101-Data.csv',usecols=COLS)
df2020 = df2020.iloc[1:]
df2020['S0101_C01_026E'] = df2020['S0101_C01_026E'].astype('int')

df2019 = pd.read_csv(path+'ACSST5Y2019.S0101-Data.csv',usecols=COLS)
df2019 = df2019.iloc[1:]
df2019['S0101_C01_026E'] = df2019['S0101_C01_026E'].astype('int')

df2018 = pd.read_csv(path+'ACSST5Y2018.S0101-Data.csv',usecols=COLS)
df2018 = df2018.iloc[1:]
df2018['S0101_C01_026E'] = df2018['S0101_C01_026E'].astype('int')

df2017 = pd.read_csv(path+'ACSST5Y2017.S0101-Data.csv',usecols=COLS)
df2017 = df2017.iloc[1:]
df2017['S0101_C01_026E'] = df2017['S0101_C01_026E'].astype('int')

df2021['cfips'] = df2021.GEO_ID.apply(lambda x: int(x.split('US')[-1]) )
adult2021 = df2021.set_index('cfips').S0101_C01_026E.to_dict()

df2020['cfips'] = df2020.GEO_ID.apply(lambda x: int(x.split('US')[-1]) )
adult2020 = df2020.set_index('cfips').S0101_C01_026E.to_dict()

df2019['cfips'] = df2019.GEO_ID.apply(lambda x: int(x.split('US')[-1]) )
adult2019 = df2019.set_index('cfips').S0101_C01_026E.to_dict()

df2018['cfips'] = df2018.GEO_ID.apply(lambda x: int(x.split('US')[-1]) )
adult2018 = df2018.set_index('cfips').S0101_C01_026E.to_dict()

df2017['cfips'] = df2017.GEO_ID.apply(lambda x: int(x.split('US')[-1]) )
adult2017 = df2017.set_index('cfips').S0101_C01_026E.to_dict()

train_data['adult2021'] = train_data.cfips.map(adult2021)
train_data['adult2020'] = train_data.cfips.map(adult2020)
train_data['adult2019'] = train_data.cfips.map(adult2019)
train_data['adult2018'] = train_data.cfips.map(adult2018)
train_data['adult2017'] = train_data.cfips.map(adult2017)

train_data["Date"] = pd.to_datetime(train_data["first_day_of_month"],format="%Y-%m-%d")
train_data["Year"] = train_data["Date"].dt.year

train_data.loc[train_data.Year==2022,"microbusiness_density"] = train_data.loc[train_data.Year==2022,"microbusiness_density"] * train_data.loc[train_data.Year==2022,"adult2020"] / train_data.loc[train_data.Year==2022,"adult2021"]
train_data.loc[train_data.Year==2021,"microbusiness_density"] = train_data.loc[train_data.Year==2021,"microbusiness_density"] * train_data.loc[train_data.Year==2021,"adult2019"] / train_data.loc[train_data.Year==2021,"adult2021"]
train_data.loc[train_data.Year==2020,"microbusiness_density"] = train_data.loc[train_data.Year==2020,"microbusiness_density"] * train_data.loc[train_data.Year==2020,"adult2018"] / train_data.loc[train_data.Year==2020,"adult2021"]
train_data.loc[train_data.Year==2019,"microbusiness_density"] = train_data.loc[train_data.Year==2019,"microbusiness_density"] * train_data.loc[train_data.Year==2019,"adult2017"] / train_data.loc[train_data.Year==2019,"adult2021"]

train_data = train_data.drop(["adult2021","adult2020","adult2019","adult2018","adult2017","Date","Year"],axis=1)

train_data.to_csv(out_path+"train_N+1.csv",index=False)
test_data.to_csv(out_path+"test_N+1.csv",index=False)

#%%
# N+2
train_data = pd.concat([train_data,test_data.loc[test_data["first_day_of_month"]=="2023-01-01"]])
train_data = train_data.sort_values(by=["cfips","row_id"])
train_data = train_data.reset_index(drop=True)
train_data["county"] = train_data.groupby(by=["cfips"],as_index=False)["county"].ffill()
train_data["state"] = train_data.groupby(by=["cfips"],as_index=False)["state"].ffill()


test_data = test_data.loc[~(test_data["first_day_of_month"]=="2023-01-01")]

train_data.to_csv(out_path+"train_N+2.csv",index=False)
test_data.to_csv(out_path+"test_N+2.csv",index=False)
#%%
# N+3
train_data = pd.concat([train_data,test_data.loc[test_data["first_day_of_month"]=="2023-02-01"]])
train_data = train_data.sort_values(by=["cfips","row_id"])
train_data = train_data.reset_index(drop=True)
train_data["county"] = train_data.groupby(by=["cfips"],as_index=False)["county"].ffill()
train_data["state"] = train_data.groupby(by=["cfips"],as_index=False)["state"].ffill()


test_data = test_data.loc[~(test_data["first_day_of_month"]=="2023-02-01")]

train_data.to_csv(out_path+"train_N+3.csv",index=False)
test_data.to_csv(out_path+"test_N+3.csv",index=False)

#%%
# N+4
train_data = pd.concat([train_data,test_data.loc[test_data["first_day_of_month"]=="2023-03-01"]])
train_data = train_data.sort_values(by=["cfips","row_id"])
train_data = train_data.reset_index(drop=True)
train_data["county"] = train_data.groupby(by=["cfips"],as_index=False)["county"].ffill()
train_data["state"] = train_data.groupby(by=["cfips"],as_index=False)["state"].ffill()


test_data = test_data.loc[~(test_data["first_day_of_month"]=="2023-03-01")]

train_data.to_csv(out_path+"train_N+4.csv",index=False)
test_data.to_csv(out_path+"test_N+4.csv",index=False)


#%%
# N+5
train_data = pd.concat([train_data,test_data.loc[test_data["first_day_of_month"]=="2023-04-01"]])
train_data = train_data.sort_values(by=["cfips","row_id"])
train_data = train_data.reset_index(drop=True)
train_data["county"] = train_data.groupby(by=["cfips"],as_index=False)["county"].ffill()
train_data["state"] = train_data.groupby(by=["cfips"],as_index=False)["state"].ffill()


test_data = test_data.loc[~(test_data["first_day_of_month"]=="2023-04-01")]

train_data.to_csv(out_path+"train_N+5.csv",index=False)
test_data.to_csv(out_path+"test_N+5.csv",index=False)


#%%
# N+6
train_data = pd.concat([train_data,test_data.loc[test_data["first_day_of_month"]=="2023-05-01"]])
train_data = train_data.sort_values(by=["cfips","row_id"])
train_data = train_data.reset_index(drop=True)
train_data["county"] = train_data.groupby(by=["cfips"],as_index=False)["county"].ffill()
train_data["state"] = train_data.groupby(by=["cfips"],as_index=False)["state"].ffill()


test_data = test_data.loc[~(test_data["first_day_of_month"]=="2023-05-01")]

train_data.to_csv(out_path+"train_N+6.csv",index=False)
test_data.to_csv(out_path+"test_N+6.csv",index=False)



