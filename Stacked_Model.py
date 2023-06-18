import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cat

path = "F:/Kaggle/microbusiness_density/February/Data/"
out_path = "F:/Kaggle/microbusiness_density/February/Final_Submission/"

#%%
# Functions

def smape(y_true, y_pred):
    smap = np.zeros(len(y_true))
    
    num = np.abs(y_true - y_pred)
    dem = ((np.abs(y_true) + np.abs(y_pred)) / 2)
    
    pos_ind = (y_true!=0)|(y_pred!=0)
    smap[pos_ind] = num[pos_ind] / dem[pos_ind]
    
    return 100 * np.mean(smap)

def smape_vector(y_true, y_pred):
    smape = np.zeros(len(y_true))
    
    numinator = np.abs(y_true - y_pred)
    denominator = ((np.abs(y_true) + np.abs(y_pred)) / 2)

    pos_ind = (y_true != 0) | (y_pred != 0)
    smape[pos_ind] = numinator[pos_ind] / denominator[pos_ind]
    
    return 100 * smape

#%%
# Data

horizon = 6
data = pd.read_csv(out_path+f"All_Features_Final_{horizon}.csv")

data = data.rename(columns={"Target":"target"})
raw = data.copy()
#all_feats = data.copy()

#%%
# Baseline Model
train = pd.read_csv(out_path+f"train_N+{horizon}.csv")
test = pd.read_csv(path+"test.csv")

for lag in [-1, 1]:
    train[f'microbusiness_density_lag_{lag}'] = train.groupby('cfips')['microbusiness_density'].shift(lag)

if horizon == 1:
    train_data = train[(train.first_day_of_month >= '2022-09-01') & (train.first_day_of_month <= '2022-11-01')]
elif horizon == 2:
    train_data = train[(train.first_day_of_month >= '2022-10-01') & (train.first_day_of_month <= '2022-12-01')]
elif horizon == 3:
    train_data = train[(train.first_day_of_month >= '2022-11-01') & (train.first_day_of_month <= '2023-01-01')]
elif horizon == 4:
    train_data = train[(train.first_day_of_month >= '2022-12-01') & (train.first_day_of_month <= '2023-02-01')]
elif horizon == 5:
    train_data = train[(train.first_day_of_month >= '2023-01-01') & (train.first_day_of_month <= '2023-03-01')]
elif horizon == 6:
    train_data = train[(train.first_day_of_month >= '2023-02-01') & (train.first_day_of_month <= '2023-04-01')]

mult_column_to_mult = {f'smape_{mult}': mult for mult in [1.00, 1.005, 1.0075]}
mult_to_priority = {1: 1, 1.005: 0.4, 1.0075: 0.2}

for mult_column, mult in mult_column_to_mult.items():
    train_data['y_pred'] = train_data['microbusiness_density'] * mult
    train_data[mult_column] = smape_vector(y_true=train_data['microbusiness_density_lag_-1'],
                                           y_pred=train_data['y_pred']) * mult_to_priority[mult]
    
df_agg = train_data.groupby('cfips')[list(mult_column_to_mult.keys())].mean()
df_agg['best_mult'] = df_agg.idxmin(axis=1).map(mult_column_to_mult)

last_active_value = train.groupby('cfips', as_index=False)['active'].last().rename(columns={'active':'last_active_value'})
df_agg = df_agg.merge(last_active_value,on='cfips')
mask = df_agg['last_active_value'] < 150
df_agg.loc[mask,'best_mult'] = 1.00

cfips_to_best_mult = dict(zip(df_agg.cfips, df_agg['best_mult']))

last_value = train.groupby('cfips',as_index=False)['microbusiness_density'].last().rename(columns={'microbusiness_density': 'last_train_value'})

submission = test.merge(last_value,on='cfips')
submission['forecast_month_number'] = submission.groupby('cfips').cumcount() + 1
submission['microbusiness_density'] = submission['last_train_value'] * submission['cfips'].map(cfips_to_best_mult)
sub = submission[['row_id', 'microbusiness_density']]

#%%


features = ['state_i', 'mbd_lag_1', 'act_lag_1', 'mbd_lag_2', 'act_lag_2', 'mbd_lag_3', 
            'act_lag_3', 'mbd_lag_4', 'act_lag_4', 'mbd_lag_5', 'act_lag_5', 'mbd_lag_6', 
            'act_lag_6', 'mbd_lag_7', 'act_lag_7', 'mbd_rollmea2_1', 'mbd_rollmea4_1', 
            'mbd_rollmea6_1', 'mbd_rollmea8_1', 'mbd_rollmea10_1', 'pct_bb_2017', 'pct_bb_2018', 
            'pct_bb_2019', 'pct_bb_2020', 'pct_bb_2021', 'pct_college_2017', 'pct_college_2018', 
            'pct_college_2019', 'pct_college_2020', 'pct_college_2021', 'pct_foreign_born_2017', 
            'pct_foreign_born_2018', 'pct_foreign_born_2019', 'pct_foreign_born_2020', 
            'pct_foreign_born_2021', 'pct_it_workers_2017', 'pct_it_workers_2018', 
            'pct_it_workers_2019', 'pct_it_workers_2020', 'pct_it_workers_2021', 'median_hh_inc_2017', 
            'median_hh_inc_2018', 'median_hh_inc_2019', 'median_hh_inc_2020', 'median_hh_inc_2021', 'SUMLEV',
            'DIVISION', 'ESTIMATESBASE2020','POPESTIMATE2020', 'POPESTIMATE2021', 'NPOPCHG2020', 'NPOPCHG2021',
            'BIRTHS2020', 'BIRTHS2021', 'DEATHS2020', 'DEATHS2021', 'NATURALCHG2020', 'NATURALCHG2021', 'INTERNATIONALMIG2020',
            'INTERNATIONALMIG2021', 'DOMESTICMIG2020', 'DOMESTICMIG2021', 'NETMIG2020', 'NETMIG2021', 'RESIDUAL2020', 'RESIDUAL2021',
            'GQESTIMATESBASE2020', 'GQESTIMATES2020', 'GQESTIMATES2021', 'RBIRTH2021', 'RDEATH2021', 'RNATURALCHG2021',
            'RINTERNATIONALMIG2021', 'RDOMESTICMIG2021','RNETMIG2021', 'lng', 'lat', 'scale', 
            'rot_15_x', 'rot_15_y', 'rot_30_x', 'rot_30_y', 'rot_45_x', 'rot_45_y']


#%%
def get_model():
    cat_model = cat.CatBoostRegressor(iterations=2000,
                                      loss_function="MAPE",
                                      verbose=0,
                                      grow_policy='SymmetricTree',
                                      learning_rate=0.035,
                                      colsample_bylevel=0.8,
                                      max_depth=5,
                                      l2_leaf_reg=0.2,
                                      subsample=0.70,
                                      max_bin=4096)
    return cat_model


def base_models(): 
    
    # LGBM model
    params = {'n_iter': 300,
              'boosting_type': 'dart',
              'verbosity': -1,
              'objective': 'l1',
              'colsample_bytree': 0.8841279649367693,
              'colsample_bynode': 0.10142964450634374,
              'max_depth': 8,
              'learning_rate': 0.003647749926797374,
              'lambda_l2': 0.5,
              'num_leaves': 61,
              "seed": 42,
              'min_data_in_leaf': 213}

    lgb_model = lgb.LGBMRegressor(**params)
    
    xgb_model = xgb.XGBRegressor(objective='reg:pseudohubererror',
                                 tree_method="hist",
                                 n_estimators=795,
                                 learning_rate=0.0075,
                                 max_leaves = 17,
                                 subsample=0.50,
                                 colsample_bytree=0.50,
                                 max_bin=4096,
                                 n_jobs=2)

    cat_model = cat.CatBoostRegressor(iterations=2500,
                                      loss_function="MAPE",   
                                      verbose=0,  
                                      grow_policy='SymmetricTree',
                                      learning_rate=0.035,
                                      colsample_bylevel=0.8,
                                      max_depth=5,
                                      l2_leaf_reg=0.2,
                                      subsample=0.70,
                                      max_bin=4096)                  
    
    models = {}
    models['xgb'] = xgb_model
    models['lgbm'] = lgb_model
    models['cat'] = cat_model

    return models


ACT_THR = 150
raw['ypred_last'] = np.nan
raw['ypred'] = np.nan
raw['k'] = 1


    
TS = 39 + horizon
print(TS)

models = base_models()
model0 = get_model()
model1 = models['lgbm']
model2 = models['cat']

train_indices = (raw.istest==0) & (raw.dcount  < TS) & (raw.dcount >= 1) & (raw.lastactive>ACT_THR)
valid_indices = (raw.dcount == TS)

model0.fit(raw.loc[train_indices,features],raw.loc[train_indices,'target'].clip(-0.0016,0.0045))
model1.fit(raw.loc[train_indices,features],raw.loc[train_indices,'target'].clip(-0.0016,0.0045))
model2.fit(raw.loc[train_indices,features],raw.loc[train_indices,'target'].clip(-0.0016,0.0045))

tr_pred0 = model0.predict(raw.loc[train_indices, features])
tr_pred1 = model1.predict(raw.loc[train_indices, features])
tr_pred2 = model2.predict(raw.loc[train_indices, features])
train_preds = np.column_stack((tr_pred0, tr_pred1, tr_pred2))

meta_model = get_model() 
meta_model.fit(train_preds,raw.loc[train_indices, 'target'].clip(-0.0016,0.0045))

val_preds0 = model0.predict(raw.loc[valid_indices, features])
val_preds1 = model1.predict(raw.loc[valid_indices, features])
val_preds2 = model2.predict(raw.loc[valid_indices, features])
valid_preds = np.column_stack((val_preds0, val_preds1, val_preds2))

ypred = meta_model.predict(valid_preds)

raw.loc[valid_indices,'k_target'] = ypred
raw.loc[valid_indices,'k'] = ypred + 1.
raw.loc[valid_indices,'k'] = raw.loc[valid_indices,'k'] * raw.loc[valid_indices,'microbusiness_density']

lastval = raw.loc[raw.dcount==TS,['cfips', 'microbusiness_density']].set_index('cfips').to_dict()['microbusiness_density']
dt = raw.loc[raw.dcount==TS,['cfips','k']].set_index('cfips').to_dict()['k']

df = raw.loc[raw.dcount==(TS+1),['cfips','microbusiness_density','state','lastactive','mbd_lag_1']].reset_index(drop=True)
df['pred'] = df['cfips'].map(dt)
df['lastval'] = df['cfips'].map(lastval)

df.loc[df['lastactive']<=ACT_THR, 'pred'] = df.loc[df['lastactive']<=ACT_THR,'lastval']

raw.loc[raw.dcount==(TS+1),'ypred'] = df['pred'].values
raw.loc[raw.dcount==(TS+1),'ypred_last'] = df['lastval'].values

dt = raw.loc[raw.dcount==(TS+1),['cfips','ypred']].set_index('cfips').to_dict()['ypred']
test = raw.loc[raw.istest==1,['row_id','cfips','microbusiness_density']].copy()
test['microbusiness_density'] = test['cfips'].map(dt)

#%%

# last_mbd = all_feats.loc[(all_feats.Year==2022)&(all_feats.Month==12)]
# last_mbd = last_mbd.reset_index(drop=True)
# last_mbd = last_mbd[["Year","Month","cfips","microbusiness_density"]]
# last_mbd = last_mbd.rename(columns={"microbusiness_density":"december_mbd"})

# nov_mbd = all_feats.loc[(all_feats.Year==2022)&(all_feats.Month==11)]
# nov_mbd = nov_mbd.reset_index(drop=True)
# nov_mbd = nov_mbd[["Year","Month","cfips","microbusiness_density"]]
# nov_mbd = nov_mbd.rename(columns={"microbusiness_density":"november_mbd"})

# sub_format = pd.read_csv(path+"prev_sub_format.csv")
# sub_format["cfips"] = np.nan
# sub_format["Date"] = np.nan
# for k,v in sub_format.iterrows():
#     c = sub_format["row_id"].iloc[[k]].str.split("_",expand=True)[0].iloc[0]
#     d = sub_format["row_id"].iloc[[k]].str.split("_",expand=True)[1].iloc[0]
#     sub_format["cfips"].iloc[k] = int(c)
#     sub_format["Date"].iloc[k] = d

# sub_format["Date"] = pd.to_datetime(sub_format["Date"],format="%Y-%m-%d")
# sub_format["Year"] = sub_format["Date"].dt.year
# sub_format["Month"] = sub_format["Date"].dt.month
# sub_format = sub_format.loc[sub_format.Date<"2023-01-01"]  
# sub_format = sub_format.reset_index(drop=True)

# sub_format = sub_format.merge(last_mbd,how="left",on=["Year","Month","cfips"])
# sub_format = sub_format.merge(nov_mbd,how="left",on=["Year","Month","cfips"])
    
# sub_format.loc[(sub_format.Year==2022)&(sub_format.Month==11),"microbusiness_density"] = sub_format.loc[(sub_format.Year==2022)&(sub_format.Month==11),"november_mbd"]
# sub_format.loc[(sub_format.Year==2022)&(sub_format.Month==12),"microbusiness_density"] = sub_format.loc[(sub_format.Year==2022)&(sub_format.Month==12),"december_mbd"]

# sub_format = sub_format[["row_id","microbusiness_density"]]

#%%

test = test[['row_id','microbusiness_density']]
#test = pd.concat([sub_format,test])
test = test.sort_values(by=["row_id"])
test = test.reset_index(drop=True)

#%%
sub["cfips"] = sub["row_id"].apply(lambda x:x.split("_")[0])
sub["Date"] = sub["row_id"].apply(lambda x:x.split("_")[1])
sub["cfips"] = sub["cfips"].astype('int64')
sub = sub.loc[~((sub.Date=="2022-11-01")|(sub.Date=="2022-12-01")|
                (sub.Date=="2023-01-01")|(sub.Date=="2023-02-01")|
                (sub.Date=="2023-03-01")|(sub.Date=="2023-04-01")|(sub.Date=="2023-05-01"))]
sub = sub.drop("Date",axis=1)

column_names = ['GEO_ID','NAME','S0101_C01_026E']
df2021 = pd.read_csv(path+'ACSST5Y2021.S0101-Data.csv', usecols=column_names)
df2021 = df2021.iloc[1:]
df2021['S0101_C01_026E'] = df2021['S0101_C01_026E'].astype('int')
df2021['cfips'] = df2021.GEO_ID.apply(lambda x: int(x.split('US')[-1]))
adult2021 = df2021.set_index('cfips').S0101_C01_026E.to_dict()
sub['adult2021'] = sub['cfips'].map(adult2021)

sub = sub.sort_values(by=["row_id"])
sub = sub.reset_index(drop=True)

test["active"] = np.nan
for i,row in sub.iterrows():
    test.iat[i,1] = (0.65*test.iat[i,1] + 0.35*row["microbusiness_density"])
    test.iat[i,2] = np.round(test.iat[i,1] * row['adult2021'] / 100)
    test.iat[i,1] = np.round(test.iat[i,1] * row['adult2021'] / 100) / row['adult2021'] * 100
test.to_csv(out_path+f'baseline_{horizon}.csv', index=False)