import pandas as pd
import numpy as np
import matplotlib as matplotlib
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import plotly.graph_objects as go


SP500 = pd.read_csv('venv/lib/SP500_data_2010_to_2024_adjusted_close.csv')
SP500.index = SP500.iloc[:, 0]
SP500.index = pd.to_datetime(SP500.index, format='%Y-%m-%d')
SP500 = SP500.iloc[:, 1:]

close = SP500['Adj Close']
high = SP500.High
low = SP500.Low


def upbreak(line, refline):
    n = min(len(line), len(refline))
    line = line[-n:]
    refline = refline[-n:]
    signal = pd.Series(0, index=line.index)
    for i in range(1, len(line)):
        if all([line[i] > refline[i], line[i-1] < refline[i-1]]):
            signal[i] = 1
    return signal


def downbreak(line, refline):
    n = min(len(line), len(refline))
    line = line[-n:]
    refline = refline[-n:]
    signal = pd.Series(0, index=line.index)
    for i in range(1, len(line)):
        if all([line[i] < refline[i], line[i-1] > refline[i-1]]):
            signal[i] = 1
    return signal


def bbands(Price, period=20, times=2):
    upband = pd.Series(0.0, index=Price.index)
    midband = pd.Series(0.0, index=Price.index)
    downband = pd.Series(0.0, index=Price.index)
    sigma = pd.Series(0.0, index=Price.index)
    for i in range(period-1, len(Price)):
        midband[i] = np.nanmean(Price[i - (period-1): (i+1)])
        sigma[i] = np.nanstd(Price[i - (period-1): (i+1)])
        upband[i] = midband[i] + times*sigma[i]
        downband[i] = midband[i] - times * sigma[i]
    BBands = pd.DataFrame({'upband': upband[(period - 1):],
                           'midband': midband[(period - 1):],
                           'downband': downband[(period - 1):]})
    return BBands


def CalBollRisk(tsPrice, multiplier):
    k = len(multiplier)
    overUp = []
    belowDown = []
    BollRisk = []
    for i in range(k):
        BBands = bbands(tsPrice, 20, multiplier[i])
        a=0
        b=0
        for j in range(len(BBands)):
            tsPrice = tsPrice[-(len(BBands)):]
            if tsPrice[j] > BBands.upband[j]:
                a += 1
            elif tsPrice[j] < BBands.downband[j]:
                b += 1
            overUp.append(a)
            belowDown.append(b)
            BollRisk.append(100*(a+b)/len(tsPrice))
    return BollRisk


def perform(Price, TradSig):
    ret = Price/Price.shift(1) - 1
    tradRet = (ret*TradSig).dropna()
    ret = ret[-len(tradRet):]
    winRate = [len(ret[ret>0]) / len(ret[ret !=0]),
               len(tradRet[tradRet>0])/ len(tradRet[tradRet !=0])]
    meanWin = [np.mean(ret[ret>0]),
               np.mean(tradRet[tradRet>0])]
    meanLoss = [np.mean(ret[ret < 0]),
               np.mean(tradRet[tradRet < 0])]
    Performance = pd.DataFrame({'winRate': winRate, 'meanWin' : meanWin,
                                'meanLoss': meanLoss})
    Performance.index=['Stock', 'Trade']
    return Performance


BBands = bbands(close, 20, 2)
upbreakBB1 = upbreak(close, BBands.upband)
downbreakBB1 = downbreak(close, BBands.downband)

upBBSig1 = -upbreakBB1.shift(2)
downBBSig1 = -downbreakBB1.shift(2)

tradSignal1 = upBBSig1 + downBBSig1
tradSignal1[tradSignal1 == -0] = 0

Performance1 = perform(close, tradSignal1)


updownbbrange = BBands[['downband', 'upband']]
multiplier = [1, 1.65, 1.96, 2, 2.58]
price2010 = close['2010-01-04': '2010-12-31']
risk = CalBollRisk(price2010, multiplier)

fig = go.Figure()
fig.add_trace(go.Candlestick(x=updownbbrange.index,
                             open=SP500['Open'],
                             high=SP500['High'],
                             low=SP500['Low'],
                             close=SP500['Adj Close'],
                             name='Candlestick'))
fig.add_trace(go.Scatter(x=updownbbrange.index, y=updownbbrange.upband,
                         line=dict(color='green', width=1), name='Upper Band'))
fig.add_trace(go.Scatter(x=updownbbrange.index, y=updownbbrange.downband,
                         line=dict(color='red', width=1), name='Lower Band'))
fig.show()