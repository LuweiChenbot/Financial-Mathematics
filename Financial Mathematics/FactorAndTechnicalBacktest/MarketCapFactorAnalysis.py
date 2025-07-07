import pandas as pd
from pandas import date_range

from dateutils import get_last_trading_day, is_trade_day, get_rebalance_dates
import scipy.stats.mstats as sp
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings("ignore", message="Series.ravel is deprecated")


prices_df = pd.read_csv('venv/lib/ClosePrice.csv')
prices_df.rename(columns={'Unnamed: 0': '日期'}, inplace=True)
prices_df['日期'] = pd.to_datetime(prices_df['日期'], format='%Y%m%d')
prices_df.set_index('日期', inplace=True)

Mktcap = pd.read_csv('venv/lib/cap.csv')
Mktcap.rename(columns={'Unnamed: 0': '日期'}, inplace=True)
Mktcap['日期'] = pd.to_datetime(Mktcap['日期'], format='%Y%m%d')

Info = pd.read_csv('venv/lib/basicinfo.csv')


def winsorize_series(s):
    q = s.quantile([0.01, 0.99])
    return np.where(s < q[0.01], q[0.01], np.where(s > q[0.99], q[0.99], s))

def calculate_max_drawdown(return_series):
    peak = return_series.expanding(min_periods=1).max()
    drawdown = (return_series / peak) - 1
    return drawdown.min()


def get_daily_scoring(start_date, percentile):

    start_date = get_last_trading_day(start_date)
    start_date = pd.to_datetime(str(start_date), format='%Y%m%d')

    find_date = Mktcap[Mktcap['日期'] == start_date]

    if find_date.empty:
        print(f"No data for {start_date.strftime('%Y%m%d')}")
        pass

    df1 = pd.DataFrame()
    df1['Cap'] = find_date.iloc[0, 1:]

    if df1[df1['Cap'] >= 3e5].empty:
        print(f"No large cap stocks on {start_date.strftime('%Y%m%d')}")
        pass

    df1 = df1[df1['Cap'] >= 3e5]
    info = Info[['证券代码', '交易所', '上市日期']]
    filtered_info = info[(info['交易所'] != 'BSE')]
    merged_data = df1.merge(filtered_info, left_index=True, right_on='证券代码', how='inner')

    #print("merged data compiled successfully")

    if merged_data.empty:
        print(f"Merge failed on {start_date}")
        pass
    merged_data['Cap'] = merged_data['Cap'].astype(float)
    merged_data['Cap'].dropna(inplace=True)
    merged_data['Log_Cap'] = np.log(merged_data['Cap'])
    merged_data['Winsorized_Log_Cap'] = winsorize_series(merged_data['Log_Cap'])
    merged_data['zscored_Winsorized_Log_Cap'] = sp.zscore(merged_data['Winsorized_Log_Cap'])

    merged_data['Percentile'] = pd.qcut(merged_data['Winsorized_Log_Cap'], percentile, labels=False)
    merged_data = merged_data.drop(columns=['交易所', '上市日期'])

    grouped = merged_data.groupby('Percentile')['证券代码'].apply(list).to_dict()
    return grouped


def back_test_portfolios_month(start_date, end_date, percentile):
    # 构建日期序列，按频率采样
    trade_periods = get_rebalance_dates(start_date, end_date)
    # 缩小价格索引范围
    start_date = pd.to_datetime(start_date, format='%Y-%m-%d')
    end_date = pd.to_datetime(end_date, format='%Y-%m-%d')
    filtered_prices_df = prices_df.loc[start_date:end_date]
    # 创建空表储存收益
    cumulative_portfolio_rtn = pd.DataFrame(index=filtered_prices_df.index,
                                            columns=[f'Portfolio_{i}' for i in range(0, percentile)])
    cumulative_portfolio_rtn['Long_Short_Ratio'] = np.nan
    cumulative_portfolio_rtn.iloc[0] = 1

    for i in range(1, len(trade_periods)):
        current_date = trade_periods[i]
        last_date = trade_periods[i - 1]
        portfolios = get_daily_scoring(current_date, percentile)
        for j in date_range:
            for p in range(percentile):
                stock_list = portfolios[p]
                current_close = prices_df.loc[current_date, stock_list]
                last_close = prices_df.loc[last_date, stock_list]

                period_returns = current_close / last_close - 1
                mean_period_returns = period_returns.mean()

                last_cum_return = cumulative_portfolio_rtn.at[last_date, f'Portfolio_{p}']
                cumulative_portfolio_rtn.at[current_date, f'Portfolio_{p}'] = last_cum_return * (1 + mean_period_returns)

        short_portfolio = cumulative_portfolio_rtn.at[current_date, 'Portfolio_0']
        long_portfolio = cumulative_portfolio_rtn.at[current_date, f'Portfolio_{percentile - 1}']

        if short_portfolio != 0:
            cumulative_portfolio_rtn.at[current_date, 'Long_Short_Ratio'] = long_portfolio / short_portfolio

    return cumulative_portfolio_rtn


def initialize(start_date, end_date, percentile):
    start_date = get_last_trading_day(start_date)
    trade_periods = get_rebalance_dates(start_date, end_date)
    # 确保索引为DatetimeIndex并过滤数据
    start_date = pd.to_datetime(start_date, format='%Y%m%d')
    end_date = pd.to_datetime(end_date, format='%Y%m%d')
    filtered_prices_df = prices_df.loc[start_date:end_date]

    # 创建DataFrame来存储每个组合的累积收益，只包含过滤后的日期范围
    cumulative_portfolio_rtn = pd.DataFrame(index=filtered_prices_df.index,
                                            columns=[f'Portfolio_{i}' for i in range(percentile)])
    #cumulative_portfolio_rtn['Long_Short_Net_Returns'] = np.nan
    # 创建DataFrame来存储每个组合的持倉，只包含过滤后的日期范围
    holdings_df = pd.DataFrame(index=filtered_prices_df.index)
    cumulative_portfolio_rtn.iloc[0] = 1  # 初始化第一行的值为1

    return cumulative_portfolio_rtn, filtered_prices_df, trade_periods


def back_test_portfolios(return_df, prices_df, trade_periods, percentile):
    # 初始化上一次重平衡的组合
    last_rebalance_portfolio = get_daily_scoring(prices_df.index[0], percentile)
    last_rebalance_date = prices_df.index[0]

    for current_date in prices_df.index[1:]:
        last_date = prices_df.index[prices_df.index.get_loc(current_date) - 1]

        for p in range(percentile):
            stock_list = last_rebalance_portfolio[p]
            current_close = prices_df.loc[current_date, stock_list]
            rebalance_close = prices_df.loc[last_rebalance_date, stock_list]

            period_returns = (current_close / rebalance_close) - 1
            mean_period_returns = period_returns.mean(skipna=True)

            last_cum_return = return_df.at[last_rebalance_date, f'Portfolio_{p}']
            return_df.at[current_date, f'Portfolio_{p}'] = last_cum_return * (1 + mean_period_returns)

        long_portfolio = return_df.at[current_date, 'Portfolio_0']
        short_portfolio = return_df.at[current_date, f'Portfolio_{percentile - 1}']
        return_df.at[current_date, 'Long_Short_Ratio'] = long_portfolio / short_portfolio

        if current_date in trade_periods:
            # 如果是重平衡日，则获取新的股票列表
            new_rebalance_portfolio = get_daily_scoring(current_date, percentile)
            last_rebalance_portfolio = new_rebalance_portfolio
            last_rebalance_date = current_date
    return_df['Long_Short_Ratio'] = return_df['Long_Short_Ratio'].bfill()
    return return_df


def kpi_generator(result_df):
    KPI = pd.DataFrame()
    KPI['Annual Returns'] = result_df.pct_change().mean() * 252
    benchmark_return = KPI['Annual Returns'].mean()
    KPI['Excess Return'] = KPI['Annual Returns'] - benchmark_return
    KPI['Annual_Volatility'] = result_df.pct_change().std() * np.sqrt(252)

    risk_free_rate = 0.02  # Assuming risk-free rate of return is 2%
    KPI['Sharpe Ratio'] = (KPI['Annual Returns'] - risk_free_rate) / KPI['Annual_Volatility']

    max_drawdown = result_df.apply(calculate_max_drawdown)
    KPI['Max Drawdown'] = max_drawdown
    KPI['Calmar Ratio'] = KPI['Annual Returns'] / - KPI['Max Drawdown']
    KPI = KPI.iloc[:-1]
    return KPI


def holding_tracker(trade_periods, percentile):
    portfolio_tracking_df = pd.DataFrame(columns=['Date', 'Ticker', 'Portfolio', 'm_returns'])
    trade_periods_dates = get_rebalance_dates(20140228, 20240229)
    for i in range(len(trade_periods)):
        # Rebalancing: fetch new stock lists
        current_day = trade_periods[i]
        next_month = trade_periods_dates[i]

        portfolios = get_daily_scoring(current_day, percentile)
        rows = []
        for p in range(percentile):
            for ticker in portfolios[p]:
                current_close = prices_df.loc[current_day, ticker]
                next_month_close = prices_df.loc[next_month, ticker]
            # Track the portfolio and group assignment
                rows.append({
                    'Date': current_day,
                    'Ticker': ticker,
                    'Portfolio': f'Portfolios {p}',
                    'm_returns': (next_month_close / current_close) - 1
                })
        portfolio_tracking_df = pd.concat([portfolio_tracking_df, pd.DataFrame(rows)], ignore_index=True)
    return portfolio_tracking_df


returns_df, filtered_prices_df, trade_periods = initialize(20140130, 20240131, 5)
result = back_test_portfolios(returns_df, filtered_prices_df, trade_periods, 5)
KPI = kpi_generator(result)
holdings = holding_tracker(trade_periods, 5)
print(result)


for column in result.columns:
    plt.plot(result.index, result[column], label=column)

plt.title('Cumulative Returns by Portfolio')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend(title='Portfolio')
plt.show()

with pd.ExcelWriter('组合绩效表现_neww.xlsx', engine='xlsxwriter') as writer:
    # Write each DataFrame to a different sheet
    result.to_excel(writer, sheet_name='Net Values')
    KPI.to_excel(writer, sheet_name='Performance Metrics')
    holdings.to_excel(writer, sheet_name='Holdings')

    # Optionally, you can customize the formatting with the xlsxwriter library
    workbook = writer.book
    format1 = workbook.add_format({'num_format': '0.00%'})
    writer.sheets['Performance Metrics'].set_column('B:C', None, format1)


