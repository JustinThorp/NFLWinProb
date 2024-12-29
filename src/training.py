from datetime import datetime

import polars as pl

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score


import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader,Dataset

from model import Model

#torch.set_num_threads(8)

data = pl.scan_parquet('data/traing_data.parquet')



model_df = data.select([
    pl.col('play_id'),
    pl.col('game_id'),
    pl.col('qtr'),
    pl.col('game_seconds_remaining'),
    pl.col('result'),
    pl.when(pl.col('result') > 0).then(1).otherwise(0).alias('home_win'),
    pl.when(pl.col('posteam') == pl.col('home_team')).then(1).otherwise(0).alias('homepos'),
    pl.col('down'),
    pl.col('ydstogo'),
    pl.col('yardline_100'),
    pl.col('total_home_score'),
    pl.col('total_away_score'),
    pl.col('total_home_score').add(pl.col('total_away_score')).alias('total_score'),
    pl.col('total_home_score').sub(pl.col('total_away_score')).alias('score_diff'),
    pl.col('spread_line')
]).filter(
    pl.col('qtr') < 5
).with_columns(
    pl.col('home_win').sub(1).mul(-1).alias('away_win')
).drop_nulls().collect()#.sample(fraction=.05)

#print(model_df[['home_win','away_win']])

train_df,val_df = train_test_split(model_df,train_size=.8)

device = 'cpu'

class CustomDataset(Dataset):
    def __init__(self, dataframe, features, target,regression):
        self.features_tensor = torch.tensor(dataframe[features].values, dtype=torch.float32).to(device)
        self.target_tensor = torch.tensor(dataframe[target].values, dtype=torch.int64).to(device)
        self.reg_tensor = torch.tensor(dataframe[regression].values, dtype=torch.float32).to(device)
    def __len__(self):
        return len(self.target_tensor)

    def __getitem__(self, idx):
        X = self.features_tensor[idx]
        y1 = self.target_tensor[idx]
        y2 = self.reg_tensor[idx]
        return X, y1,y2

# Define the features and target
features = [
    'game_seconds_remaining',
    'score_diff',
    'total_score',
    'spread_line',
    'homepos',
    'down',
    'ydstogo',
    'yardline_100']
target = 'home_win'
regression = 'result'
# Create datasets
train_dataset = CustomDataset(train_df.to_pandas(), features, target,regression)
val_dataset = CustomDataset(val_df.to_pandas(), features, target,regression)



# Create data loaders
#train_loader = DataLoader(train_dataset, batch_size=64000, num_workers=8,pin_memory=True,persistent_workers=True)
train_loader = DataLoader(train_dataset, batch_size=100000, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=6400, shuffle=False)




net = Model().to(device)
opt = optim.Adam(net.parameters(), lr=.02) #.0002
loss_fn1 = nn.CrossEntropyLoss()
loss_fn2 = nn.MSELoss()

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(dim=1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def train(dataloader, model, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y1,y2) in enumerate(dataloader):
        start = datetime.now()

        # Compute prediction error
        pred1,pred2 = model(X)
        
        loss1 = loss_fn1(pred1, y1)
        
        loss2 = loss_fn2(pred2.squeeze(), y2)
        loss = .05*loss2 + loss1
        loss.backward()
        # Backpropagation
        optimizer.step()
        optimizer.zero_grad()
        #print(datetime.now() - start)
        if batch % 1 == 0:
            loss_1, current = loss1.item(), (batch + 1) * len(X)
            loss_2, current = loss2.item(), (batch + 1) * len(X)
            print(f"loss1: {loss_1:>7f}, loss2: {loss_2:>7f}  [{current:>5d}/{size:>5d}]")

if __name__ == "__main__":
    epochs = 50
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_loader, net, opt)
        #test(val_loader, net, loss_fn)
    torch.save(net.state_dict(), 'model.ph')
    print("Done!")