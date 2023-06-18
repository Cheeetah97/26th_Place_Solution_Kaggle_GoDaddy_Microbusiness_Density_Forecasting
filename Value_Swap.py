import pandas as pd
import numpy as np

path = "F:/Kaggle/microbusiness_density/February/Data/"
out_path = "F:/Kaggle/microbusiness_density/February/Final_Submission/"


horizon = 5


train = pd.read_csv(out_path+f"train_N+{horizon+1}.csv")

for i in range(1,horizon+1):
    b = pd.read_csv(out_path+f"baseline_{i}.csv")
    b["cfips"] = b["row_id"].apply(lambda x:x.split("_")[0])
    b["cfips"] = b["cfips"].astype('int64')
    
    b["Date"] = b["row_id"].apply(lambda x:x.split("_")[1])
    b = b.loc[b.Date==f"2023-0{i}-01"]
    b = b.reset_index(drop=True)

    for k,v in b.iterrows():
        train.loc[train.row_id==b.row_id.iloc[k],"microbusiness_density"] = b.microbusiness_density.iloc[k]
        train.loc[train.row_id==b.row_id.iloc[k],"active"] = b.active.iloc[k]

train.to_csv(out_path+f"train_N+{horizon+1}.csv",index=False)