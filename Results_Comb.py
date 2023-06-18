import pandas as pd
import numpy as np

path = "F:/Kaggle/microbusiness_density/February/Data/"
out_path = "F:/Kaggle/microbusiness_density/February/Final_Submission/"

comb_df = pd.DataFrame()
for i in range(1,7):
   sub = pd.read_csv(out_path+f"baseline_{i}.csv")
   sub["Date"] = sub["row_id"].apply(lambda x:x.split("_")[1])
   sub["cfips"] = sub["row_id"].apply(lambda x:x.split("_")[0])
   sub["cfips"] = sub["cfips"].astype('int64')
   sub = sub.loc[sub.Date==f"2023-0{i}-01"]
   sub = sub.reset_index(drop=True)
   
   sub2 = pd.read_csv(out_path+f"baseline_{i}_New.csv")
   sub2["Date"] = sub2["row_id"].apply(lambda x:x.split("_")[1])
   sub2["cfips"] = sub2["row_id"].apply(lambda x:x.split("_")[0])
   sub2["cfips"] = sub2["cfips"].astype('int64')
   sub2 = sub2.loc[sub2.Date==f"2023-0{i}-01"]
   sub2 = sub2.reset_index(drop=True)
   sub2 = sub2.rename(columns={"microbusiness_density":"microbusiness_density_new"})
   
   sub = sub.merge(sub2[["cfips","row_id","microbusiness_density_new"]])
   comb_df = pd.concat([comb_df,sub])
   
comb_df = comb_df.rename(columns={"Date":"first_day_of_month"})
comb_df["Portion"] = "future"
comb_df["microbusiness_density_new"] = (comb_df["microbusiness_density_new"] + comb_df["microbusiness_density"])/2 

train = pd.read_csv(out_path+"train_N+1.csv")
train["Portion"] = "Actual"
train["microbusiness_density_new"] = train["microbusiness_density"].copy()
train = pd.concat([train,comb_df])

train = train.drop(["county","state"],axis=1)
train.to_csv(out_path+"CHECK.csv",index=False)


#%%

sub_high = train[["first_day_of_month","row_id","cfips","microbusiness_density"]]
sub_high = sub_high.loc[sub_high.first_day_of_month.isin(["2022-11-01","2022-12-01","2023-01-01",
                                                          "2023-02-01","2023-03-01","2023-04-01",
                                                          "2023-05-01","2023-06-01"])]
sub_high = sub_high.reset_index(drop=True)
sub_high[["row_id","microbusiness_density"]].to_csv(out_path+"high_submission.csv",index=False)

#%%

sub_low = train[["first_day_of_month","row_id","cfips","microbusiness_density_new"]]
sub_low = sub_low.loc[sub_low.first_day_of_month.isin(["2022-11-01","2022-12-01","2023-01-01",
                                                          "2023-02-01","2023-03-01","2023-04-01",
                                                          "2023-05-01","2023-06-01"])]
sub_low = sub_low.reset_index(drop=True)
sub_low = sub_low.rename(columns={"microbusiness_density_new":"microbusiness_density"})
sub_low[["row_id","microbusiness_density"]].to_csv(out_path+"Low_submission.csv",index=False)

   