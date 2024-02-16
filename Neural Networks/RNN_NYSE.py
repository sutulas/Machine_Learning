# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 09:39:14 2023

@author: sutul
"""

# Fit RNN to the NYSE data


from dl_imports import *

def main():
    NYSE = load_data('NYSE')
    
    # converting day_of_week to integer values
    NYSE['day_of_week'] = NYSE['day_of_week'].rank(
        method='dense', ascending=False).astype(int) 
    
    # adding day_of_week
    cols = ['DJ_return', 'log_volume', 'log_volatility', 'day_of_week']
    X = pd.DataFrame(StandardScaler(
        with_mean=True, 
        with_std=True).fit_transform(NYSE[cols]), 
        columns=NYSE[cols].columns, index=NYSE.index)
 
    for lag in range(1, 6):
        for col in cols:
            newcol = np.zeros(X.shape[0]) * np.nan
            newcol[lag:] = X[col].values[:-lag]
            X.insert(len(X.columns), "{0}_{1}".format(col, lag), newcol)
    X.insert(len(X.columns), 'train', NYSE['train'])
    X = X.dropna()
    
    Y, train = X['log_volume'], X['train']
    X = X.drop(columns=['train'] + cols)
    
    X = pd.merge(
        X, pd.get_dummies(NYSE['day_of_week']),on='date')
    
    X.columns = X.columns.astype(str)

    ordered_cols = []
    for lag in range(5,0,-1):
        for col in cols:
            ordered_cols.append('{0}_{1}'.format(col, lag))
    X = X.reindex(columns=ordered_cols)
    
    # change shape so it fits the addition of day_of_week
    X_rnn = X.to_numpy().reshape((-1,5,4)) 
    
    class NYSEModel(nn.Module):
        def __init__(self):
            super(NYSEModel, self).__init__()
            # there are now 4 inputs, not 3
            self.rnn = nn.RNN(4, 12, batch_first=True) 
            self.dense = nn.Linear(12, 1)
            self.dropout = nn.Dropout(0.1)
        def forward(self, x):
            val, h_n = self.rnn(x)
            val = self.dense(self.dropout(val[:,-1]))
            return torch.flatten(val)
    
    nyse_model = NYSEModel()
    
    datasets = []
    for mask in [train, ~train]:
        X_rnn_t = torch.tensor(X_rnn[mask].astype(np.float32))
        Y_t = torch.tensor(Y[mask].astype(np.float32))
        datasets.append(TensorDataset(X_rnn_t, Y_t))
    nyse_train, nyse_test = datasets
    
    print(summary(
            nyse_model, input_data=X_rnn_t, 
            col_names=['input_size','output_size','num_params'])
            )
    
    nyse_dm = SimpleDataModule(
        nyse_train, nyse_test, 
        num_workers=min(4, max_num_workers),
        validation=nyse_test,batch_size=64
        )
    
    for idx, (x, y) in enumerate(nyse_dm.train_dataloader()):
        out = nyse_model(x)
        print(y.size(), out.size())
        if idx >= 2:
            break
        
    nyse_optimizer = RMSprop(nyse_model.parameters(), lr=0.001)
    nyse_module = SimpleModule.regression(
        nyse_model, optimizer=nyse_optimizer, 
        metrics={'r2':R2Score()})
    
    nyse_trainer = Trainer(
        deterministic=True, max_epochs=200, 
        callbacks=[ErrorTracker()])
    nyse_trainer.fit(nyse_module, datamodule=nyse_dm)
    print(nyse_trainer.test(nyse_module,datamodule=nyse_dm))
    
if __name__ == "__main__":
    main()
    
    
    
                        #################
                        #    results    #
                        #################
    
    # 50 epochs:
    # ┌───────────────────────────┬───────────────────────────┐
    # │        Test metric        │       DataLoader 0        │
    # ├───────────────────────────┼───────────────────────────┤
    # │         test_loss         │    0.5754546523094177     │
    # │          test_r2          │    0.4538654685020447     │
    # └───────────────────────────┴───────────────────────────┘
    
    # 200 epochs:
    # ┌───────────────────────────┬───────────────────────────┐
    # │        Test metric        │       DataLoader 0        │
    # ├───────────────────────────┼───────────────────────────┤
    # │         test_loss         │    0.5742294192314148     │
    # │          test_r2          │     0.455028235912323     │
    # └───────────────────────────┴───────────────────────────┘
    
    # the orginal R2, without day_of_week, is 0.4172, with 200 epochs
    # as we can see in the above results, adding the extra variable improves
    # the R2, even with just 50 epochs
    # the R2 increase between 50 and 200 epochs isn't very much and probably
    # is not worth the extra computation time
