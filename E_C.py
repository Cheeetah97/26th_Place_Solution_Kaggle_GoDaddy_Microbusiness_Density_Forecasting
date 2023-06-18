from deap import algorithms, base, creator, tools
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
random.seed(42)
np.random.seed(42)

#%%
path = "F:/Kaggle/microbusiness_density/February/Data/"

data = pd.read_csv(path+"opt2.csv")
sub = pd.read_csv(path+"baseline_noconversion.csv")

all_feats = pd.read_csv(path+"All_Features3.csv")

last_mbd = all_feats.loc[(all_feats.Year==2022)&(all_feats.Month==12)]
last_mbd = last_mbd.reset_index(drop=True)
last_mbd = last_mbd[["Year","Month","cfips","microbusiness_density"]]
last_mbd = last_mbd.rename(columns={"microbusiness_density":"december_mbd"})

#%%
# RMSE Calculation
def rmse(targets,predictions):
    return np.sqrt(((targets - predictions)**2).mean())

#%%

def fitness_function(all_preds,true_values,solution):
    # calculate the mean prediction for each sample
    mean_preds = np.mean(all_preds, axis=0)
    # use the solution as a binary mask to select the predictions to use
    selected_preds = all_preds[solution, :]
    # calculate the mean prediction for each sample using the selected predictions
    mean_selected_preds = np.mean(selected_preds, axis=0)
    # calculate the mean squared error between the mean selected predictions and the true values
    mse = np.mean((mean_selected_preds - true_values)**2)
    # calculate the mean squared error between the mean predictions and the true values
    mse_all = np.mean((mean_preds - true_values)**2)
    # return the negative difference between the two mse values as the fitness (maximize fitness = minimize difference)
    return - (mse_all - mse)

def tournament_selection(population,fitness_scores,tournament_size=2):
    # select a random subset of the population
    tournament = random.sample(population,tournament_size)
    # calculate the fitness scores for the solutions in the tournament
    tournament_fitness = [fitness_scores[population.index(solution)] for solution in tournament]
    # select the solution with the highest fitness score
    winner = tournament[np.argmax(tournament_fitness)]
    return winner

def crossover(parent_1, parent_2):
    # select a random crossover point
    crossover_point = random.randint(0, len(parent_1) - 1)
    # combine the parent solutions at the crossover point
    offspring = np.concatenate((parent_1[:crossover_point], parent_2[crossover_point:]))
    return [int(x) for x in list(offspring)]

def mutate(solution, mutation_rate):
    # make a copy of the solution to avoid modifying the input array
    mutation_points = np.where(np.random.rand(len(solution)) < mutation_rate)[0]
    mutation_values = np.random.normal(0,mutation_rate,size=len(mutation_points))
    mutated_solution = solution.copy()
    for i in range(len(mutation_points)):
        index = mutation_points[i]
        mutated_solution[index] = int(mutated_solution[index] + mutation_values[i])
    return mutated_solution


# define the genetic algorithm parameters
population_size = 100
mutation_rate = 0.05
elite_size = 5
generations = 50

final_df = pd.DataFrame()
for cnty in list(data.cfips.unique()):
    temp = data.loc[data.cfips==cnty]
    temp = temp.reset_index(drop=True)
    
    valid_data = temp.loc[temp.Portion=='Test']
    future_data = temp.loc[temp.Portion=='future']
    
    valid_data = valid_data.reset_index(drop=True)
    valid_data = valid_data.drop("Portion",axis=1)
    
    future_data = future_data.reset_index(drop=True)
    future_data = future_data.drop("Portion",axis=1)
    
    all_preds = np.array([valid_data["pred_ps_1"].tolist(),valid_data["pred_ps_2"].tolist(),valid_data["pred_ps_3"].tolist(),valid_data["pred_ps_4"].tolist()])
    true_values = valid_data["Target"].tolist()
    future_preds = np.array([future_data["pred_ps_1"].tolist(),future_data["pred_ps_2"].tolist(),future_data["pred_ps_3"].tolist(),future_data["pred_ps_4"].tolist()])
    
    # define the initial population
    population = []
    for i in range(population_size):
        solution = [random.randint(0,1) for i in range(4)]
        population.append(solution)
    
    # define the main loop
    for i in range(generations):
        # calculate the fitness for each solution in the population
        fitness_scores = [fitness_function(all_preds,true_values,solution) for solution in population]
        # select the elite solutions (the best solutions from the previous generation)
        elite_indices = np.argsort(fitness_scores)[-elite_size:]
        elite_solutions = [population[idx] for idx in elite_indices]
        # initialize the new population with the elite solutions
        new_population = elite_solutions.copy()
        # fill the rest of the new population with new solutions created by crossover and mutation
        while len(new_population) < population_size:
            # select two parent solutions by tournament selection
            parent1 = tournament_selection(population, fitness_scores)
            parent2 = tournament_selection(population, fitness_scores)
            # create a new solution by crossover
            child = crossover(parent1,parent2)
            # mutate the child solution
            child = mutate(child, mutation_rate)
            #print(child)
            # add the child solution to the new population
            new_population.append(child)
            
    # replace the old population with the new population
    population = new_population

    # select the best solution (the solution with the highest fitness)
    #best_solution = max(population,key=lambda x: fitness_function(solution=x,true_values=true_values,all_preds=all_preds))
    best_solution = max(population, key=lambda x: np.mean(all_preds[x, :]))
    # use the best solution as a binary mask to select the predictions to use
    selected_preds = all_preds[best_solution, :]

    # calculate the mean prediction for each sample
    mean_preds = np.mean(selected_preds, axis=0)
    
    # the final predictions are the mean predictions
    final_preds = mean_preds.tolist()
    
    #%%
    # Future Predictions

    # Use the best solution to select the predictions to use and calculate the mean prediction for each sample
    selected_future_preds = future_preds[best_solution, :]
    mean_future_preds = np.mean(selected_future_preds, axis=0)
    
    # The final predictions for the future data are the mean predictions
    final_future_preds = mean_future_preds.tolist()
    
    #%%
    
    final_preds = final_preds + final_future_preds

    plt.plot(temp["Target"])
    plt.plot(temp["pred_ps_1"],label="model")
    plt.plot(temp["pred_ps_2"],label="kaggle")
    plt.plot(temp["pred_ps_3"],label="3")
    plt.plot(temp["pred_ps_4"],label="4")
    plt.plot(final_preds,label="ec")
    plt.title(cnty)
    plt.legend()
    plt.show()
    
    temp.loc[temp.Portion=="future","Target"] = final_future_preds[0]
    final_df = pd.concat([final_df,temp])
    
    
future_df = final_df.loc[final_df.Portion=="future"]
future_df = future_df.reset_index(drop=True)

future_df = future_df.merge(last_mbd[["cfips","december_mbd"]],how="left",on=["cfips"])
future_df["jan_pred_ec"] = (future_df["Target"]+1)*future_df["december_mbd"]
future_df["jan_pred_kaggle"] = (future_df["pred_ps_2"]+1)*future_df["december_mbd"]
future_df["jan_pred_model"] = (future_df["pred_ps_1"]+1)*future_df["december_mbd"]
future_df = future_df.reset_index(drop=True)


best_skus = [10005,1011,1105,12131,13113,13285,16021,17135,17171,
             17197,21165,21181,21215,23017,26081,28035,32017,40143,
             41061,42003,46085,47187,48155,5095,51051,53065,56033,
             6087,6115,8033]

# best_skus = [1011,13285,16021,17135,17171,
#               17197,21165,21181,23017,40143,
#               42003,46085,47187,48155,5095,53065,56033,
#               6087,8033]

best_results = pd.read_csv(path+"Best_Results2.csv")
best_results = best_results.loc[best_results.Portion=="future"]
best_results = best_results.reset_index(drop=True)
#%%
# Kaggle Upload

for k,v in sub.iterrows():
    c = sub["row_id"].iloc[[k]].str.split("_",expand=True)[0].iloc[0]
    
    #if (int(c) in best_skus) & (int(c) in list(future_df.cfips.unique())):
        #print(int(c))
        #sub["microbusiness_density"].iloc[k] = (best_results.loc[best_results.cfips==int(c),"pred"].iloc[0] + 1)*(last_mbd.loc[last_mbd.cfips==int(c),"december_mbd"].iloc[0])
    #    sub["microbusiness_density"].iloc[k] = future_df.loc[future_df.cfips==int(c),"jan_pred_model"].iloc[0]
    if int(c) in list(future_df.cfips.unique()):
        sub["microbusiness_density"].iloc[k] = future_df.loc[future_df.cfips==int(c),"jan_pred_ec"].iloc[0]


COLS = ['GEO_ID','NAME','S0101_C01_026E']
df2020 = pd.read_csv(path+'ACSST5Y2020.S0101-Data.csv',usecols=COLS)
df2020 = df2020.iloc[1:]
df2020['S0101_C01_026E'] = df2020['S0101_C01_026E'].astype('int')

df2021 = pd.read_csv(path+'ACSST5Y2021.S0101-Data.csv',usecols=COLS)
df2021 = df2021.iloc[1:]
df2021['S0101_C01_026E'] = df2021['S0101_C01_026E'].astype('int')

sub['cfips'] = sub.row_id.apply(lambda x:int(x.split('_')[0]))

df2020['cfips'] = df2020.GEO_ID.apply(lambda x: int(x.split('US')[-1]) )
adult2020 = df2020.set_index('cfips').S0101_C01_026E.to_dict()

df2021['cfips'] = df2021.GEO_ID.apply(lambda x: int(x.split('US')[-1]) )
adult2021 = df2021.set_index('cfips').S0101_C01_026E.to_dict()

sub['adult2020'] = sub.cfips.map(adult2020)
sub['adult2021'] = sub.cfips.map(adult2021)

sub.microbusiness_density = sub.microbusiness_density * sub.adult2020 / sub.adult2021
sub = sub.drop(['adult2020','adult2021','cfips'],axis=1)
sub.to_csv(path+"sub_new.csv",index=False)
