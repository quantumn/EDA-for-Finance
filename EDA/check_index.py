#!/usr/bin/env python
# coding: utf-8

# ### 先物指数や株価指数等を取得して描画し、相関を確認
# ####  https://note.com/scilabcafe/n/n91f21c5aa65b

import pandas_datareader.data as web
import yfinance as yf
import datetime
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import gridspec
import japanize_matplotlib

# 確認対象銘柄

# stooqより取得
codelists_stooq = [
    '^DJIA', # S&P 500
]

# yahoofinanceより取得
codelists_yf = [
    'HG=F', # copper(銅)
    'GC=F', # Gold(金)
    'CL=F', # Oil(原油)
    'SOXX', # 半導体
    '^DJT', # Transportation Average Index(DJTA)
    '^RUT', # Russel 2000
    '^VIX', # VIX index
    '^SKEW', # SKEW index
    ## CBOE SKEW Index（CBOEスキュー指数）市場参加者の長期的なリスクに関する見通しを提供する指数。
    ## 高いスキュー値は、市場の大きな下落のリスクが増加していることを示唆
    'HYG', # ジャンクボンド
    '^TNX', # アメリカ10年国債
    'DX=F', # ドルインデックス
    'BTC-USD', # BTC
    'ETH-USD', # ETH
]

# 開始・終了日の設定
start = datetime.date.today() - datetime.timedelta(days=180)
end = datetime.date.today()

# データ取得(stooq)
_df_stooq = web.DataReader(codelists_stooq, 'stooq', start, end)['Close']

# 日付を昇順に並び替える
_df_stooq.sort_index(inplace=True)

# データ取得(yahoo finance)
_df_yahoo = yf.download(codelists_yf, start, end)['Adj Close']

# _df_stooqと_df_yahooを結合する
df = pd.DataFrame()
df = pd.merge(_df_stooq, _df_yahoo, on='Date', how='left')

# カラム名変更
df.rename(columns={
    '^SPX':'S&P 500',
    'BTC-USD':'BTC',
    'CL=F':'原油',
    'DX=F':'ドル',
    'ETH-USD':'ETH',
    'GC=F':'金',
    'HG=F':'銅',
    'HYG':'ジャンクボンド',
    'SOXX':'半導体',
    '^DJT':'Dow30',
    '^RUT':'Rs2000',
    '^SKEW':'SKEW',
    '^TNX':'金利10Y',
    '^VIX':'VIX',
               
},inplace=True)

df.head(5)


##取得したデータを一覧表示するおまけ機能###
df.plot(figsize=(14, 16), linewidth=2, alpha=0.5, subplots=True, layout=(7,3), grid=False)
plt.show()

import plotly.graph_objects as go # グラフ表示関連ライブラリ
import plotly.io as pio # 入出力関連ライブラリ
# pio.renderers.default = 'iframe'
from plotly import offline

from plotly.subplots import make_subplots  # subplot


# グラフの実体となる trace オブジェクトを生成
trace_1 = go.Scatter(x=df.index, y=df['S&P 500'], mode='lines', line=dict(color='red', width=2), name='SP500')
trace_2 = go.Scatter(x=df.index, y=df['ドル'], mode='lines', line=dict(color='red', width=2), name='ドル')
trace_3 = go.Scatter(x=df.index, y=df['BTC'], mode='lines', line=dict(color='red', width=2), name='BTC')        
trace_4 = go.Scatter(x=df.index, y=df['ETH'], mode='lines', line=dict(color='red', width=2), name='ETH')
trace_5 = go.Scatter(x=df.index, y=df['ジャンクボンド'], mode='lines', line=dict(color='red', width=2), name='ジャンクボンド')
trace_6 = go.Scatter(x=df.index, y=df['金'], mode='lines', line=dict(color='red', width=2), name='金')
trace_7 = go.Scatter(x=df.index, y=df['銅'], mode='lines', line=dict(color='red', width=2), name='銅')
trace_8 = go.Scatter(x=df.index, y=df['原油'], mode='lines', line=dict(color='red', width=2), name='原油')
trace_9 = go.Scatter(x=df.index, y=df['半導体'], mode='lines', line=dict(color='red', width=2), name='半導体')
trace_10 = go.Scatter(x=df.index, y=df['Rs2000'], mode='lines', line=dict(color='red', width=2), name='Rs2000')
trace_11 = go.Scatter(x=df.index, y=df['SKEW'], mode='lines', line=dict(color='red', width=2), name='SKEW')
trace_12 = go.Scatter(x=df.index, y=df['VIX'], mode='lines', line=dict(color='red', width=2), name='VIX')
trace_13 = go.Scatter(x=df.index, y=df['Dow30'], mode='lines', line=dict(color='red', width=2), name='Dow30')
trace_14 = go.Scatter(x=df.index, y=df['金利10Y'], mode='lines', line=dict(color='red', width=2), name='金利10Y')


rows=4
cols=4

subplots_fig = make_subplots(
    rows=rows,
    cols=cols,
    start_cell='top-left',
    subplot_titles=[
        'SP500',
        'ドル',
        'BTC',
        'ETH',
        'ジャンクボンド',
        '金',
        '銅',
        '原油',
        '半導体',
        'Rs2000',
        'SKEW',
        'VIX',
        'Transportation Average Index',
        '米国10年国債',
    ],
    horizontal_spacing=0.08,
    vertical_spacing=0.12,
)


# 描画領域である figure オブジェクトの作成                  
subplots_fig.add_trace(trace_1, row=1, col=1)
subplots_fig.add_trace(trace_2, row=1, col=2)
subplots_fig.add_trace(trace_3, row=1, col=3)
subplots_fig.add_trace(trace_4, row=1, col=4)

subplots_fig.add_trace(trace_5, row=2, col=1)
subplots_fig.add_trace(trace_6, row=2, col=2)
subplots_fig.add_trace(trace_7, row=2, col=3)
subplots_fig.add_trace(trace_8, row=2, col=4)

subplots_fig.add_trace(trace_9, row=3, col=1)
subplots_fig.add_trace(trace_10, row=3, col=2)
subplots_fig.add_trace(trace_11, row=3, col=3)
subplots_fig.add_trace(trace_12, row=3, col=4)

subplots_fig.add_trace(trace_13, row=4, col=1)
subplots_fig.add_trace(trace_14, row=4, col=2)

# レイアウトの更新
subplots_fig.update_layout(
    
    # 凡例は表示
    showlegend=False,
    
    # 幅と高さの設定
    width=1200,height=900,
    
    title='米国の主要インデックスと各種指数',
    plot_bgcolor='white', # 背景色を白に設定
    
),


for row in range(1,rows+1):
    for col in range(1, cols+1):
        
        # linecolorを設定して、ラインをミラーリング（mirror=True）して枠にする
        subplots_fig.update_xaxes(linecolor='black', linewidth=1, mirror=True, row=row, col=col)
        subplots_fig.update_yaxes(linecolor='black', linewidth=1, mirror=True, row=row, col=col)
        
        # ticks='inside'：目盛り内側, tickcolor：目盛りの色, tickwidth：目盛りの幅、ticklen：目盛りの長さ
        subplots_fig.update_xaxes(ticks='inside', tickcolor='black', tickwidth=1, ticklen=5, row=row, col=col)
        subplots_fig.update_yaxes(ticks='inside', tickcolor='black', tickwidth=1, ticklen=5, row=row, col=col)
        
        # gridcolor：グリッドの色, gridwidth：グリッドの幅、griddash='dot'：破線
        subplots_fig.update_xaxes(gridcolor='lightgrey', gridwidth=1, griddash='dot', row=row, col=col)
        subplots_fig.update_yaxes(gridcolor='lightgrey', gridwidth=1, griddash='dot', row=row, col=col)

        # 軸の文字サイズ変更
        subplots_fig.update_xaxes(tickfont=dict(size=12, color='grey'), row=row, col=col)
        subplots_fig.update_yaxes(tickfont=dict(size=14, color='grey'), row=row, col=col)
        
# show()メソッドでグラフを描画
# subplots_fig.show()

# ファイルサイズを小さくしたい場合にはTrueを選べば良い
offline.plot(subplots_fig, filename = 'basic-line', auto_open = True)

df.corr() # 相関関係

import seaborn as sns

plt.figure(figsize=(14, 9)) 
cor = df.corr()
sns.heatmap(cor, cmap= sns.color_palette('coolwarm', 10), annot=True,fmt='.2f', vmin = -1, vmax = 1)
plt.show()





