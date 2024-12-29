import polars as pl
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from model import Model



data = pl.scan_parquet('data/traing_data.parquet')

net = Model()
net.load_state_dict(torch.load('model.ph', weights_only=True))
net.eval()

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
features = [
    'game_seconds_remaining',
    'score_diff',
    'total_score',
    'spread_line',
    'homepos',
    'down',
    'ydstogo',
    'yardline_100']
X = model_df[features]
X = torch.tensor(X.to_numpy(),dtype=torch.float32)
preds = net(X)[0]#.detach().numpy()
preds2 = net(X)[1]#.detach().numpy()


model_df = model_df.with_columns(
    pl.lit(F.softmax(preds,dim = 1)[:,1].detach().numpy()).alias('homewinprob'),
    pl.lit(preds2[:,0].detach().numpy()).alias('homewinmargin')
).sort('game_seconds_remaining',descending=True).select([
    pl.col('game_id'),
    pl.col('qtr'),
    pl.col('game_seconds_remaining'),
    pl.col('home_win'),
    pl.col('homewinprob')
]).with_columns([
    pl.col('homewinprob').rolling_mean(2,min_periods=1,weights=[.25,.75]).over('game_id').alias('rollingwinprob')
]).group_by('qtr').agg(
    pl.map_groups(['home_win','rollingwinprob'],lambda x: roc_auc_score(x[0],x[1]))
).sort('qtr')



print(model_df)