import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer, precision_score, recall_score, accuracy_score, f1_score,roc_auc_score, balanced_accuracy_score
import optuna

from MEDfl.LearningManager.model import Model
from MEDfl.LearningManager.strategy import Strategy
from MEDfl.LearningManager.server import FlowerServer
from MEDfl.LearningManager.flpipeline import FLpipeline

class BinaryClassifier(nn.Module):
    def __init__(self, input_size, num_layers, layer_size):
        super(BinaryClassifier, self).__init__()

        # Input layer
        self.layers = [nn.Linear(input_size, layer_size)]
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(layer_size, layer_size))
        
        # Output layer
        self.layers.append(nn.Linear(layer_size, 1))
        
        # ModuleList to handle dynamic number of layers
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return x
    
class CustomPyTorchClassifier(BaseEstimator):
    def __init__(self, hidden_dim=10, lr=0.001, pos_weight=1, th=0.5, max_epochs=10, batch_size=32):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.pos_weight = pos_weight
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.th = th
        self.model = None

    def fit(self, X, y):
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        if isinstance(y, torch.Tensor):
            y = y.numpy()

        input_dim = X.shape[1]
        self.model = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )

        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.pos_weight))
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        train_data = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.max_epochs):
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        if isinstance(X, torch.Tensor):
            X = X.numpy()

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(torch.from_numpy(X).float())
            predictions = (outputs.squeeze() > self.th).float().numpy()
        return predictions

    def score(self, X, y):
        predictions = self.predict(X)
        return accuracy_score(y, predictions)


class ParamsOptimiser:
    def __init__(self, X_train = None, y_train=None, X_test=None, y_test=None):
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.to_numpy()
        if isinstance(y_train, pd.Series):
            y_train = y_train.to_numpy()
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.to_numpy()
        if isinstance(y_test, pd.Series):
            y_test = y_test.to_numpy()

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def perform_grid_search(self, param_grid, scoring_metric='recall', cv=3, verbose=1):
        pytorch_model = CustomPyTorchClassifier()
        scorer = make_scorer(recall_score, greater_is_better=True)

        if scoring_metric == 'precision':
            scorer = make_scorer(precision_score)
        elif scoring_metric == 'accuracy':
            scorer = make_scorer(accuracy_score)
        elif scoring_metric == 'recall':
            scorer = make_scorer(recall_score)
        elif scoring_metric == 'f1':
            scorer = make_scorer(f1_score)

        grid_search = GridSearchCV(pytorch_model, param_grid, scoring=scorer, cv=cv, refit=scoring_metric, verbose=verbose)
        grid_search.fit(self.X_train, self.y_train)

        self.grid_search_results = grid_search  # Save the grid search results

        return grid_search
    
    # Inside the CustomModelTrainer class
    def plot_results(self, params_to_plot=None):
        results = pd.DataFrame(self.grid_search_results.cv_results_)

        if params_to_plot is None:
            # Create a column for configuration details
            results['config'] = results['params'].apply(lambda x: str(x))

            # Visualize mean test scores along with configurations
            plt.figure(figsize=(15, 8))
            bar_plot = plt.bar(results.index, results['mean_test_score'], color='blue', alpha=0.7)
            plt.xticks(results.index, results['config'], rotation='vertical', fontsize=8)
            plt.ylabel('Mean Test Score')
            plt.title('Mean Test Scores for Each Configuration')
            plt.tight_layout()

            # Add values on top of bars
            for bar, score in zip(bar_plot, results['mean_test_score']):
                plt.text(bar.get_x() + bar.get_width() / 2 - 0.15, bar.get_height() + 0.01, f'{score:.3f}', fontsize=8)

            plt.show()
            return 

        try:
            # Dynamically get the column names for the specified scoring metric
            mean_test_col = f'mean_test_{params_to_plot[0]}'
            param_cols = [f'param_{param}' for param in params_to_plot]

            if len(params_to_plot) == 1:
                # Plotting the heatmap for a single parameter
                plt.figure(figsize=(8, 6))
                sns.heatmap(results.pivot_table(index=param_cols[0]),
                            annot=True, cmap='YlGnBu', fmt=".3f", cbar_kws={'label': mean_test_col})
                plt.title(mean_test_col.capitalize())
                plt.show()
            elif len(params_to_plot) == 2:
                # Create a pair plot for two parameters
                plt.figure(figsize=(8, 6))
                scores = results.pivot_table(index=param_cols[0], columns=param_cols[1], values=f'mean_test_score', aggfunc="mean")
                sns.heatmap(scores, annot=True, cmap='YlGnBu', fmt=".3f", cbar_kws={'label': mean_test_col})
                plt.title(mean_test_col.capitalize())
                plt.show()
            else:
                print("Invalid number of parameters to plot. You can provide either one or two parameters.")
        except KeyError as e:
            print(f"Error: {e}. Make sure the specified scoring metric exists in the results DataFrame.")

    

    def optuna_optimisation(self, direction, params):
        # Create the data loaders here
        train_data = TensorDataset(torch.from_numpy(self.X_train).float(), torch.from_numpy(self.y_train).float())
        test_data = TensorDataset(torch.from_numpy(self.X_test).float(), torch.from_numpy(self.y_test).float())

       

        def objective(trial):

            batch_size=trial.suggest_int('batch_size', **params['batch_size'])

            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

            # Create the model with the suggested hyperparameters
            model = BinaryClassifier(input_size=self.X_train.shape[1],
                                     num_layers=trial.suggest_int('num_layers', **params['num_layers']) ,
                                     layer_size=trial.suggest_int('hidden_size', **params['hidden_size']))

            # Define the loss function and optimizer
            criterion = nn.BCEWithLogitsLoss()
            optimizer_name = trial.suggest_categorical('optimizer', params['optimizer'])
            learning_rate = trial.suggest_float('learning_rate', **params['learning_rate'])
            

            if optimizer_name == 'Adam':
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            elif optimizer_name == 'SGD':
                optimizer = optim.SGD(model.parameters(), lr=learning_rate)
            elif optimizer_name == 'RMSprop':
                optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

            # Training loop
            num_epochs = trial.suggest_int('num_epochs', **params['num_epochs'])
            for epoch in range(num_epochs):
                model.train()
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    loss.backward()
                    optimizer.step()

                # Evaluation
                model.eval()
                predictions = []
                true_labels = []
                with torch.no_grad():
                    for batch_X, batch_y in test_loader:
                        outputs = model(batch_X)
                        predictions.extend(torch.sigmoid(outputs).numpy())
                        true_labels.extend(batch_y.numpy())

                # Calculate F1 score
                # f1 = f1_score(true_labels, (np.array(predictions) > 0.5).astype(int))
                auc = roc_auc_score(true_labels, predictions)

                trial.report(auc, epoch)

                # Handle pruning based on the intermediate value
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return auc

        # Create an Optuna study
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=params['n_trials'])
        
        self.study = study

        # Get the best hyperparameters
        best_params = study.best_params
        print(f"Best Hyperparameters: {best_params}")

        return study
    
    def train_optimized_model(self ,trial ,th_min , th_max):
        
        best_params = self.study.best_params

        threshold = trial.suggest_float('threashhold', th_min, th_max, log=True)

        train_data = TensorDataset(torch.from_numpy(self.X_train).float(), torch.from_numpy(self.y_train).float())
        test_data = TensorDataset(torch.from_numpy(self.X_test).float(), torch.from_numpy(self.y_test).float())

        train_loader = DataLoader(train_data, batch_size=best_params['batch_size'], shuffle=True)
        test_loader = DataLoader(test_data, batch_size=best_params['batch_size'], shuffle=False)

        
        # Use the best hyperparameters to train the final model
        final_model = BinaryClassifier(input_size=self.X_train.shape[1], layer_size=best_params['hidden_size'] , num_layers=best_params['num_layers'])
        final_optimizer = self.get_optimizer(best_params['optimizer'], final_model.parameters(), best_params['learning_rate'])
        final_criterion = nn.BCEWithLogitsLoss()

        num_epochs = best_params['num_epochs']
        for epoch in range(num_epochs):
            final_model.train()
            for batch_X, batch_y in train_loader:
                final_optimizer.zero_grad()
                outputs = final_model(batch_X)
                loss = final_criterion(outputs.squeeze(), batch_y)
                loss.backward()
                final_optimizer.step()

        # Evaluate the final model on the test set
        final_model.eval()
        with torch.no_grad():
            predictions = []
            true_labels = []
            for batch_X, batch_y in test_loader:
                outputs = final_model(batch_X)
                predictions.extend(torch.sigmoid(outputs).numpy())
                true_labels.extend(batch_y.numpy())

        final_balanced_acc = balanced_accuracy_score(true_labels, (np.array(predictions) > threshold).astype(int))
        print(f"Model balanced accuracy: {final_balanced_acc}")

        return final_balanced_acc

    def get_optimizer(self, optimizer_name, parameters, learning_rate):
        if optimizer_name == 'Adam':
            return optim.Adam(parameters, lr=learning_rate)
        elif optimizer_name == 'SGD':
            return optim.SGD(parameters, lr=learning_rate)
        elif optimizer_name == 'RMSprop':
            return optim.RMSprop(parameters, lr=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def perform_grid_search(self, param_grid, scoring_metric='recall', cv=3, verbose=1):
        pytorch_model = CustomPyTorchClassifier()
        scorer = make_scorer(recall_score, greater_is_better=True)

        if scoring_metric == 'precision':
            scorer = make_scorer(precision_score)
        elif scoring_metric == 'accuracy':
            scorer = make_scorer(accuracy_score)
        elif scoring_metric == 'recall':
            scorer = make_scorer(recall_score)
        elif scoring_metric == 'f1':
            scorer = make_scorer(f1_score)

        grid_search = GridSearchCV(pytorch_model, param_grid, scoring=scorer, cv=cv, refit=scoring_metric, verbose=verbose)
        grid_search.fit(self.X_train, self.y_train)

        self.grid_search_results = grid_search  # Save the grid search results

        return grid_search
    
    
    def plot_param_importances(self):
        return optuna.visualization.plot_param_importances(self.study)
    
    def plot_slice(self , params):
        return optuna.visualization.plot_slice(self.study , params=params)
    
    def plot_parallel_coordinate(self):
        return optuna.visualization.plot_parallel_coordinate(self.study)
    
    def plot_rank(self , params=None):
        return optuna.visualization.plot_rank(self.study , params=params)
    
    def plot_optimization_history(self):
        return optuna.visualization.plot_optimization_history(self.study)
    
    def optimize_model_threashhold(self , n_trials , th_min , th_max):
        additional_params = {'th_min': th_min, 'th_max': th_max}

        th_study = optuna.create_study(direction='maximize')
        th_study.optimize(lambda trial: self.train_optimized_model(trial , **additional_params) , n_trials)

        # Get the best hyperparameters
        best_params = th_study.best_params
        print(f"Best Hyperparameters: {best_params}")

        return optuna.visualization.plot_rank(th_study , params=['threashhold'])
    
    def federated_params_iptim(self , params , direction,  model, fl_dataset):

        def objective(trial):
          
            criterion = nn.BCEWithLogitsLoss()

            optimizer_name = trial.suggest_categorical('optimizer', params['optimizer'])
            learning_rate = trial.suggest_float('learning_rate', **params['learning_rate'])
            num_rounds = trial.suggest_int('num_rounds', **params['num_rounds'])
            diff_privacy =  trial.suggest_int('diff_privacy', **params['diff_privacy'])
            diff_privacy = True if diff_privacy == 1 else False

            if optimizer_name == 'Adam':
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            elif optimizer_name == 'SGD':
                optimizer = optim.SGD(model.parameters(), lr=learning_rate)
            elif optimizer_name == 'RMSprop':
                optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

            # Creating a new Model instance using the specific model created by DynamicModel
            global_model = Model(model, optimizer, criterion)

            # Get the initial params of the model 
            init_params = global_model.get_parameters() 

            fl_strategy = trial.suggest_categorical('fl_strategy', params['fl_strategy'])

            learning_strategy = Strategy(fl_strategy, 
                   fraction_fit = 1.0 ,
                   fraction_evaluate = 1.0,
                   min_fit_clients = 2,
                   min_evaluate_clients = 2,
                   min_available_clients = 2 , 
                   initial_parameters=init_params)
            
            learning_strategy.create_strategy()

            # Create The server 
            server = FlowerServer(global_model, strategy = learning_strategy, num_rounds = num_rounds,
                       num_clients  = len(fl_dataset.trainloaders), 
                       fed_dataset = fl_dataset,diff_privacy = diff_privacy,
                       # You can change the resources alocated for each client based on your machine 
                       client_resources={'num_cpus': 1.0, 'num_gpus': 0.0}
                       )
            
            ppl_1 = FLpipeline( name ="the first fl_pipeline",description = "this is our first FL pipeline",
                   server = server)
            
            # Run the Traning of the model
            history = ppl_1.server.run()

            return server.auc[len(server.auc)-1]
        

       
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=params['n_trials'])
        
        self.study = study

        # Get the best hyperparameters
        best_params = study.best_params
        print(f"Best Hyperparameters: {best_params}")

        return study


        








        
    




