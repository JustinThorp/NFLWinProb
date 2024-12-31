import polars as pl
import torch
import torch.nn.functional as F
from model import Model
import matplotlib.pyplot as plt


GAMEID = '2023_17_DET_DAL'

net = Model()
net.load_state_dict(torch.load('model.ph', weights_only=True))
net.eval()
features = [
    'game_seconds_remaining',
    'score_diff',
    'total_score',
    'spread_line',
    'homepos',
    'down',
    'ydstogo',
    'yardline_100']
net(torch.tensor([[30*60,20,20,3,1,1,10,45]],dtype = torch.float32))

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
).filter(
    pl.col('game_id') == GAMEID
).with_columns(
    pl.col('home_win').sub(1).mul(-1).alias('away_win')
).drop_nulls().collect()#.sample(fraction=.05)
X = model_df[features]
X = torch.tensor(X.to_numpy(),dtype=torch.float32)
preds = net(X)[0]#.detach().numpy()
preds2 = net(X)[1]#.detach().numpy()
fig,ax = plt.subplots(figsize = (9,6))
plot_df = model_df.with_columns(
    pl.col('game_seconds_remaining').mul(-1),
    pl.lit(F.softmax(preds,dim = 1)[:,2].detach().numpy()).alias('homewinprob'),
    pl.lit(preds2[:,0].detach().numpy()).alias('homewinmargin')
).to_pandas()
plot_df.plot(x='game_seconds_remaining',y = 'homewinprob',ax=ax,label = 'Home Team Win Prob')
xticks = []
for i in range(0,5):
    ax.axvline(-(3600 - i*15*60),linestyle = 'dashed',color = 'black',alpha = .2)
    xticks.append(-(3600 - i*15*60))
ax2 = ax.twinx()
plot_df.plot(x='game_seconds_remaining',y = 'homewinmargin',ax=ax2,color = 'orange',label = 'Home Team Margin')
ax.set_xticks(xticks)
ax.set_xticklabels(['Begin','End Q1','End Q2','End Q3','End Q4'])
ax.set_xlabel('')
ax.set_ylabel('Home Win Prob')
ax2.set_ylabel('Home win Margin')
ax.axhline(.5,linestyle = 'dashed',color = 'black',alpha = .05)
#ax2.axhline(0,linestyle = 'dashed',color = 'black',alpha = .05)
ax.set_ylim(0,1)
ax2.set_ylim(-30,30)
plt.show()