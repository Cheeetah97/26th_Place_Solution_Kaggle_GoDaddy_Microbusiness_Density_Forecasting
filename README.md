## GoDaddy_Microbusiness_Density_Forecasting
This was a Kaggle Competition. The goal of this competition was to predict monthly microbusiness densities of US counties. This repository contains that code that ranks 30th on the Private leaderboard of the competition.

The approach was as follows:

### Data Preprocessing
Since the objective of the competition was to forecast the densities for the first half of 2023, the effect of population change didn't matter in my opinion. As such, I recalculated the densities for all the years using 2021 population. (p.s in this competition the densities were calculated using population data from two years back).

### Features and Models
The features used were:

- Target lags ( 1-7)
- Active businesses difference (1-7)
- Target Lag 1 Rolling Mean window (2-10)
- Exogenous features(census, lat & long)

The models used were:

- LGBM
- XGB
- Catboost

and the results were stacked using another catboost model with less trees to avoid overfitting.

![stacking](https://github.com/Cheeetah97/GoDaddy_Microbusiness_Density_Forecasting/assets/62606459/196b9cf1-293d-4433-b536-b93f01add07b)

Post Processing
I used Linear Regression in conjunction with Rolling Median to identify the counties with the highest consitent rate of the change. Among the counties having a popuation greater than 15000, these counties were the ones I was more interested in. Using the trends of the last 2-4 months, I adjusted the predictions by multiplying them with a carefully trend factor.

In the end it seemed to have worked! Here are some of the adjusted counties.

![image](https://github.com/Cheeetah97/GoDaddy_Microbusiness_Density_Forecasting/assets/62606459/b7e8b87f-5772-446a-aad9-81e7dff09326)

![image](https://github.com/Cheeetah97/GoDaddy_Microbusiness_Density_Forecasting/assets/62606459/9a36857e-48bf-4be7-85c0-597b068a3f7f)


