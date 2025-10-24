#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas_datareader.data as web
import datetime
import pandas as pd

# 日本の主要インデックス
codelists_1 = [
    "^NKX", # 日経平均株価
    "^TPX", # TOPIX
]

# TOPIX-17の業種別ETF証券コード
codelists_2 = [
    "1617", # NF・食品(TPX17) ETF
    "1618", # NF・エネルギー資源(TPX17) ETF
    "1619", # NF・建設・資材(TPX17) ETF
    "1620", # NF・素材・化学(TPX17) ETF
    "1621", # NF・医療品(TPX17) ETF
    "1622", # NF・自動車・輸送機(TPX17) ETF
    "1623", # NF・鉄鋼・非鉄(TPX17) ETF
    "1624", # NF・機械(TPX17) ETF
    "1625", # NF・電機・精密(TPX17) ETF
    "1626", # NF・情報・サービス他(TPX17) ETF
    "1627", # NF・電力・ガス(TPX17) ETF
    "1628", # NF・運輸・物流(TPX17) ETF
    "1629", # NF・商社・卸売(TPX17) ETF
    "1630", # NF・小売(TPX17) ETF
    "1631", # NF・銀行(TPX17) ETF
    "1632", # NF・金融(TPX17) ETF
    "1633", # NF・不動産(TPX17) ETF
]


# In[2]:


_code = []
for codelist in codelists_1:
    tmp = codelist
    _code.append(tmp)

for codelist in codelists_2:
    tmp = codelist + '.JP'   
    _code.append(tmp)

# 開始・終了日の設定
start = datetime.date.today() - datetime.timedelta(days=180)
end = datetime.date.today()

df = web.DataReader(_code, 'stooq', start, end)['Close']

# カラム名変更
df.rename(columns={
    '^NKX':'日経平均株価',
    '^TPX':'TOPIX',
    '1617.JP':'TOPIX-17食品',
    '1618.JP':'TOPIX-17エネルギー資源',
    '1619.JP':'TOPIX-17建設・資材',
    '1620.JP':'TOPIX-17素材・化学',
    '1621.JP':'TOPIX-17医薬品',
    '1622.JP':'TOPIX-17自動車・輸送機',
    '1623.JP':'TOPIX-17鉄鋼・非鉄',
    '1624.JP':'TOPIX-17機械',
    '1625.JP':'TOPIX-17電機・精密',
    '1626.JP':'TOPIX-17情報通信・サービスその他',
    '1627.JP':'TOPIX-17電力・ガス',
    '1628.JP':'TOPIX-17運輸・物流',
    '1629.JP':'TOPIX-17商社・卸売',
    '1630.JP':'TOPIX-17小売',
    '1631.JP':'TOPIX-17銀行',
    '1632.JP':'TOPIX-17金融(除く銀行)',
    '1633.JP':'TOPIX-17不動産',
},inplace=True)


# In[3]:


import plotly.graph_objects as go  # グラフ表示関連ライブラリ
import plotly.io as pio  # 入出力関連ライブラリ
pio.renderers.default = 'iframe'

# subplot
from plotly.subplots import make_subplots


# In[4]:


# グラフの実体となる trace オブジェクトを生成
trace_nkx = go.Scatter(x=df.index, y=df['日経平均株価'], mode='lines', line=dict(color='red', width=2), name='日経平均株価')
trace_tpx = go.Scatter(x=df.index, y=df['TOPIX'], mode='lines', line=dict(color='red', width=2), name='TOPIX')
trace_1617 = go.Scatter(x=df.index, y=df['TOPIX-17食品'], mode='lines', line=dict(color='red', width=2), name='TOPIX-17食品')
trace_1618 = go.Scatter(x=df.index, y=df['TOPIX-17エネルギー資源'], mode='lines', line=dict(color='red', width=2), name='TOPIX-17エネルギー資源')
trace_1619 = go.Scatter(x=df.index, y=df['TOPIX-17建設・資材'], mode='lines', line=dict(color='red', width=2), name='TOPIX-17建設・資材')
trace_1620 = go.Scatter(x=df.index, y=df['TOPIX-17素材・化学'], mode='lines', line=dict(color='red', width=2), name='TOPIX-17素材・化学')
trace_1621 = go.Scatter(x=df.index, y=df['TOPIX-17医薬品'], mode='lines', line=dict(color='red', width=2), name='TOPIX-17医薬品')
trace_1622 = go.Scatter(x=df.index, y=df['TOPIX-17自動車・輸送機'], mode='lines', line=dict(color='red', width=2), name='TOPIX-17自動車・輸送機')
trace_1623 = go.Scatter(x=df.index, y=df['TOPIX-17鉄鋼・非鉄'], mode='lines', line=dict(color='red', width=2), name='TOPIX-17鉄鋼・非鉄')
trace_1624 = go.Scatter(x=df.index, y=df['TOPIX-17機械'], mode='lines', line=dict(color='red', width=2), name='TOPIX-17機械')
trace_1625 = go.Scatter(x=df.index, y=df['TOPIX-17電機・精密'], mode='lines', line=dict(color='red', width=2), name='TOPIX-17電機・精密')
trace_1626 = go.Scatter(x=df.index, y=df['TOPIX-17情報通信・サービスその他'], mode='lines', line=dict(color='red', width=2), name='TOPIX-17情報通信・サービスその他')
trace_1627 = go.Scatter(x=df.index, y=df['TOPIX-17電力・ガス'], mode='lines', line=dict(color='red', width=2), name='TOPIX-17電力・ガス')
trace_1628 = go.Scatter(x=df.index, y=df['TOPIX-17運輸・物流'], mode='lines', line=dict(color='red', width=2), name='TOPIX-17運輸・物流')
trace_1629 = go.Scatter(x=df.index, y=df['TOPIX-17商社・卸売'], mode='lines', line=dict(color='red', width=2), name='TOPIX-17商社・卸売')
trace_1630 = go.Scatter(x=df.index, y=df['TOPIX-17小売'], mode='lines', line=dict(color='red', width=2), name='TOPIX-17小売')
trace_1631 = go.Scatter(x=df.index, y=df['TOPIX-17銀行'], mode='lines', line=dict(color='red', width=2), name='TOPIX-17銀行')
trace_1632 = go.Scatter(x=df.index, y=df['TOPIX-17金融(除く銀行)'], mode='lines', line=dict(color='red', width=2), name='TOPIX-17金融(除く銀行)')
trace_1633 = go.Scatter(x=df.index, y=df['TOPIX-17不動産'], mode='lines', line=dict(color='red', width=2), name='TOPIX-17不動産')


# In[5]:


subplots_fig = make_subplots(
    rows=5,
    cols=4,
    start_cell='top-left',
    subplot_titles=[
        '日経平均株価',
        'TOPIX',
        'TOPIX-17食品',
        'TOPIX-17エネルギー資源',
        'TOPIX-17建設・資材',
        'TOPIX-17素材・化学',
        'TOPIX-17医薬品',
        'TOPIX-17自動車・輸送機',
        'TOPIX-17鉄鋼・非鉄',
        'TOPIX-17機械',
        'TOPIX-17電機・精密',
        'TOPIX-17情報通信・サービスその他',
        'TOPIX-17電力・ガス',
        'TOPIX-17運輸・物流',
        'TOPIX-17商社・卸売',
        'TOPIX-17小売',
        'TOPIX-17銀行',
        'TOPIX-17金融(除く銀行)',
        'TOPIX-17不動産',
    ],
    horizontal_spacing=0.08,
    vertical_spacing=0.12,

)


# In[6]:


# 描画領域である figure オブジェクトの作成                  
subplots_fig.add_trace(trace_nkx, row=1, col=1)
subplots_fig.add_trace(trace_tpx, row=1, col=2)
subplots_fig.add_trace(trace_1617, row=1, col=3)
subplots_fig.add_trace(trace_1618, row=1, col=4)
subplots_fig.add_trace(trace_1619, row=2, col=1)
subplots_fig.add_trace(trace_1620, row=2, col=2)
subplots_fig.add_trace(trace_1621, row=2, col=3)
subplots_fig.add_trace(trace_1622, row=2, col=4)
subplots_fig.add_trace(trace_1623, row=3, col=1)
subplots_fig.add_trace(trace_1624, row=3, col=2)
subplots_fig.add_trace(trace_1625, row=3, col=3)
subplots_fig.add_trace(trace_1626, row=3, col=4)
subplots_fig.add_trace(trace_1627, row=4, col=1)
subplots_fig.add_trace(trace_1628, row=4, col=2)
subplots_fig.add_trace(trace_1629, row=4, col=3)
subplots_fig.add_trace(trace_1630, row=4, col=4)
subplots_fig.add_trace(trace_1631, row=5, col=1)
subplots_fig.add_trace(trace_1632, row=5, col=2)
subplots_fig.add_trace(trace_1633, row=5, col=3)


# In[7]:


# レイアウトの更新
subplots_fig.update_layout(
    
    # 凡例は表示
    showlegend=False,
    
    # 幅と高さの設定
    width=1200,height=900,
    
    title='日本の主要インデックスと業種別ETF(TOPIX-17)',
    plot_bgcolor='white', # 背景色を白に設定
    
)
    


# In[11]:


rows=5
cols=4
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
subplots_fig.show()


# In[19]:


df_5=df.pct_change(5,fill_method=None)*100
df_20=df.pct_change(20,fill_method=None)*100
df_60=df.pct_change(60,fill_method=None)*100


# In[21]:


df_20


# In[22]:


import plotly.graph_objects as go  # グラフ表示関連ライブラリ
import plotly.io as pio  # 入出力関連ライブラリ
pio.renderers.default = 'iframe'

# subplot
from plotly.subplots import make_subplots

# グラフの実体となる trace オブジェクトを生成
trace_5 = go.Bar(
    x=df_5.iloc[-1],
    y=df_5.columns,
    orientation='h',
    name='現在値から5営業日前',
    
    marker=dict(
        color='blue',  # 棒自体の色
        line=dict(color='blue', width=1.5),  # 枠線の色
    ),
    
)

trace_20 = go.Bar(
    x=df_20.iloc[-1],
    y=df_20.columns,
    orientation='h',
    name='現在値から20営業日前',
    
    marker=dict(
        color='rgb(158,202,225)',  # 棒自体の色
        line=dict(color='rgb(8,48,107)', width=1.5),  # 枠線の色
    ),
    
    opacity=0.8,  # 棒の不透明度
    
)

trace_60 = go.Bar(
    x=df_60.iloc[-1],
    y=df_60.columns,
    orientation='h',
    name='現在値から60営業日前-17食品',

    marker=dict(
        color='rgb(100,202,100)',  # 棒自体の色
        line=dict(color='rgb(8,48,107)', width=1.5),  # 枠線の色
    ),
    
    opacity=0.6,  # 棒の不透明度
)


subplots_fig = make_subplots(
    rows=1,
    cols=2,
    start_cell='top-left',
    subplot_titles=[
        '現在から5営業日前 or 20営業日前',
        '現在から60営業日前',
    ],
    horizontal_spacing=0.35,
    vertical_spacing=0.12,

)


# 描画領域である figure オブジェクトの作成                  
subplots_fig.add_trace(trace_20, row=1, col=1)
subplots_fig.add_trace(trace_5, row=1, col=1)
subplots_fig.add_trace(trace_60, row=1, col=2)


# レイアウトの更新
subplots_fig.update_layout(
    
    # 凡例は表示
    showlegend=True,
    
    # 幅と高さの設定
    width=1200,height=900,
    
    title='直近のパフォーマンス（リターン）',
    plot_bgcolor='white', # 背景色を白に設定
    
    # 各バーを重ね書き
    barmode='overlay',
    
),
    
for row in range(1,rows+1):
    for col in range(1, cols+1):
        
        # linecolorを設定して、ラインをミラーリング（mirror=True）して枠にする
        subplots_fig.update_xaxes(linecolor='black', linewidth=1, mirror = True, row=row, col=col)
        subplots_fig.update_yaxes(linecolor='black', linewidth=1, mirror=True, row=row, col=col)
        
        # ticks='inside'：目盛り内側, tickcolor：目盛りの色, tickwidth：目盛りの幅、ticklen：目盛りの長さ
        subplots_fig.update_xaxes(ticks='inside', tickcolor='black', tickwidth=1, ticklen=5, row=row, col=col)
        subplots_fig.update_yaxes(ticks='inside', tickcolor='black', tickwidth=1, ticklen=5, row=row, col=col)
        
        # gridcolor：グリッドの色, gridwidth：グリッドの幅、griddash='dot'：破線
        subplots_fig.update_xaxes(gridcolor='lightgrey', gridwidth=1, griddash='dot', row=row, col=col)
        subplots_fig.update_yaxes(gridcolor='lightgrey', gridwidth=1, griddash='dot', row=row, col=col)

        # 軸の文字サイズ変更
        subplots_fig.update_xaxes(tickfont=dict(size=16, color='grey'), row=row, col=col)
        subplots_fig.update_yaxes(tickfont=dict(size=14, color='grey'), row=row, col=col)

        # 軸のタイトル
        subplots_fig.update_xaxes(title=dict(text='リターン[%]',font=dict(color='grey', size=16)))
        
        subplots_fig.update_xaxes(range=(-10, 30) ,tick0=-10, dtick=5)
        
# show()メソッドでグラフを描画
subplots_fig.show()


# In[ ]:




