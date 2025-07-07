import pandas as pd
from dateutils import get_last_trading_day, is_trade_day, get_rebalance_dates
import numpy as np
import warnings
import statsmodels.api as sm
warnings.simplefilter(action='ignore', category=FutureWarning)


# Data Reading
market_cap = pd.read_csv('venv/lib/cap.csv')  # SMB
market_cap.rename(columns={'Unnamed: 0': '日期'}, inplace=True)

bp_ratio = pd.read_csv('venv/lib/bp.csv')  # HML
bp_ratio.rename(columns={'Unnamed: 0': '日期'}, inplace=True)

index_portfolios = pd.read_csv('venv/lib/指数成分股权重_中证全指.csv')  # UMD

full_prices_df = pd.read_csv('venv/lib/ClosePrice.csv')
full_prices_df.rename(columns={'Unnamed: 0': '日期'}, inplace=True)
full_prices_df.set_index('日期', inplace=True)

hs_300 = pd.read_csv('venv/lib/HS300.csv')
hs_300.set_index('date', inplace=True)
hs_300.index = pd.to_datetime(hs_300.index)
hs_300.index = hs_300.index.strftime('%Y%m%d').astype(int)


def price_filtering(start_date, end_date):
    filtered_prices_df = full_prices_df.loc[start_date:end_date]
    return filtered_prices_df


def get_monthly_scoring(start_date):  # Typically at the last trading day of a month
    # repositioning the starting date to ensure it is a trading day
    df = pd.DataFrame()
    start_date = get_last_trading_day(start_date)

    daily_index_list = index_portfolios[index_portfolios['截止日期'] == start_date]
    df['Tickers'] = daily_index_list[['成分股代码']]
    ticker_list = daily_index_list['成分股代码'].tolist()  # 考虑放入权重

    filtered_market_cap = market_cap[market_cap['日期'] == start_date]
    filtered_bp_ratio = bp_ratio[bp_ratio['日期'] == start_date]
    filtered_market_cap.set_index('日期', inplace=True)
    filtered_bp_ratio.set_index('日期', inplace=True)

    valid_ticker_list = [ticker for ticker in ticker_list
                         if ticker in filtered_market_cap.columns
                         and ticker in filtered_bp_ratio.columns]

    # Fetching market cap and pb ratio for corresponding stocks in portfolio
    found_market_cap = filtered_market_cap.loc[start_date, valid_ticker_list]
    found_bp_ratio = filtered_bp_ratio.loc[start_date, valid_ticker_list]

    combined_data = pd.DataFrame({
        'Market_Cap': found_market_cap,
        'BP_Ratio': found_bp_ratio})
    combined_data = combined_data.dropna()

    # Processing fetched data
    combined_data['Market_Cap'] = np.log(combined_data['Market_Cap'])
    cap_categories = pd.qcut(combined_data['Market_Cap'], 3, labels=False)
    bp_ratio_categories = pd.qcut(combined_data['BP_Ratio'], 3, labels=False)

    grouped_data = pd.DataFrame({
        'Tickers': combined_data.index,
        'Market_Cap': cap_categories,
        'BP_Ratio': bp_ratio_categories})
    grouped_data = grouped_data.dropna()

    integrate_df = grouped_data

    conditions = [
        (grouped_data['Market_Cap'] == 0) & (grouped_data['BP_Ratio'] == 2),
        (grouped_data['Market_Cap'] == 0) & (grouped_data['BP_Ratio'] == 1),
        (grouped_data['Market_Cap'] == 0) & (grouped_data['BP_Ratio'] == 0),
        (grouped_data['Market_Cap'] == 2) & (grouped_data['BP_Ratio'] == 2),
        (grouped_data['Market_Cap'] == 2) & (grouped_data['BP_Ratio'] == 1),
        (grouped_data['Market_Cap'] == 2) & (grouped_data['BP_Ratio'] == 0),
    ]
    choices = ['S/H', 'S/M', 'S/L', 'B/H', 'B/M', 'B/L']

    integrate_df['Group'] = np.select(conditions, choices, default='Other')

    grouped = integrate_df.groupby('Group')
    group_dicts = {k: v['Tickers'].tolist() for k, v in grouped}
    return group_dicts


def momentum_scoring(start_date, percentile=10):
    # repositioning the starting date to ensure it is a trading day
    df = pd.DataFrame()
    start_date = get_last_trading_day(start_date)
    daily_index_list = index_portfolios[index_portfolios['截止日期'] == start_date]
    df['Tickers'] = daily_index_list[['成分股代码']]
    ticker_list = daily_index_list['成分股代码'].tolist()  # 考虑放入权重

    # Define dates for calculating period returns
    date_str = str(start_date)
    date_dt = pd.to_datetime(date_str, format='%Y%m%d')
    sakunen = date_dt - pd.DateOffset(years=1)
    sakunen_int = int(sakunen.strftime('%Y%m%d'))
    dates_range = get_rebalance_dates(sakunen_int, start_date)
    last_year = dates_range[-2]
    recent_month = dates_range[-1]

    # Calculating yearly returns
    last_year_close = full_prices_df.loc[last_year, ticker_list]
    recent_month_close = full_prices_df.loc[recent_month, ticker_list]
    yearly_return = (recent_month_close / last_year_close) - 1
    yearly_return.dropna()
    categories = pd.qcut(yearly_return, percentile, labels=False)
    filtered_cate = categories[categories.isin([0, 9])]
    return_dict = filtered_cate.groupby(filtered_cate).apply(lambda x: list(x.index)).to_dict()

    return return_dict, ticker_list

# 1 for calculating UMD/Port_return, 0 for calculating HML/SMB


def get(prices, ratio, trade_periods):
    # Initialize the return dataframe
    if ratio == 0:
        return_df = pd.DataFrame(index=prices.index,
                                 columns=['sl', 'sm', 'sh', 'bl', 'bm', 'bh', 'SMB', 'HML'])
        return_df.iloc[0] = 1

        last_rebalance_portfolio = get_monthly_scoring(trade_periods[0])
        last_rebalance_date = prices.index[0]

        for current_date in prices.index[1:]:
            sl = last_rebalance_portfolio['S/L']
            sm = last_rebalance_portfolio['S/M']
            sh = last_rebalance_portfolio['S/H']

            bl = last_rebalance_portfolio['B/L']
            bm = last_rebalance_portfolio['B/M']
            bh = last_rebalance_portfolio['B/H']

            portfolios = {
                'sl': sl,
                'sm': sm,
                'sh': sh,
                'bl': bl,
                'bm': bm,
                'bh': bh
            }
            close_prices = {}
            rebalance_close_prices = {}

            for key, stocks in portfolios.items():
                # Fetching close prices for current and last rebalance dates
                close_prices[f'{key}_close'] = prices.loc[current_date, stocks]
                rebalance_close_prices[f'{key}_rebalance_close'] = prices.loc[last_rebalance_date, stocks]

                period_returns = (close_prices[f'{key}_close'] / rebalance_close_prices[f'{key}_rebalance_close']) - 1
                mean_returns = period_returns.mean(skipna=True)

                last_cum_return = return_df.at[last_rebalance_date, f'{key}']
                return_df.at[current_date, f'{key}'] = last_cum_return * (1 + mean_returns)

            if current_date in trade_periods:
                # 如果是重平衡日，则获取新的股票列表
                new_rebalance_portfolio = get_monthly_scoring(current_date)
                last_rebalance_portfolio = new_rebalance_portfolio
                last_rebalance_date = current_date

    else:
        return_df = pd.DataFrame(index=prices.index,
                                 columns=['UMD_top', 'UMD_btm', 'Portfolio Return'])
        return_df.iloc[0] = 1
        last_rebalance_portfolio, ticker_list = momentum_scoring(trade_periods[0], percentile=10)
        last_rebalance_date = prices.index[0]

        for current_date in prices.index[1:]:
            top, btm = (last_rebalance_portfolio[9], last_rebalance_portfolio[0])

            # Fetching close prices for top
            top_close = prices.loc[current_date, top]
            top_rebalance_close = prices.loc[last_rebalance_date, top]

            # Calculating return for top
            top_period_returns = (top_close / top_rebalance_close) - 1
            top_mean_returns = top_period_returns.mean(skipna=True)
            top_last_cum_return = return_df.at[last_rebalance_date, 'UMD_top']
            return_df.at[current_date, 'UMD_top'] = top_last_cum_return * (1 + top_mean_returns)

            # Fetching close prices for bottom
            btm_close = prices.loc[current_date, btm]
            btm_rebalance_close = prices.loc[last_rebalance_date, btm]

            # Calculating return for bottom
            btm_period_returns = (btm_close / btm_rebalance_close) - 1
            btm_mean_returns = btm_period_returns.mean(skipna=True)
            btm_last_cum_return = return_df.at[last_rebalance_date, 'UMD_btm']
            return_df.at[current_date, 'UMD_btm'] = btm_last_cum_return * (1 + btm_mean_returns)

            # Calculating portfolio return
            port_close = prices.loc[current_date, ticker_list]
            port_rebalance_close = prices.loc[last_rebalance_date, ticker_list]

            port_period_returns = ((port_close / port_rebalance_close) - 1.015)
            port_mean_returns = port_period_returns.mean(skipna=True)
            port_last_cum_return = return_df.at[last_rebalance_date, 'Portfolio Return']
            return_df.at[current_date, 'Portfolio Return'] = port_last_cum_return * (1 + port_mean_returns)

            if current_date in trade_periods:
                # 如果是重平衡日，则获取新的股票列表
                new_rebalance_portfolio, ticker_list = momentum_scoring(current_date, percentile=10)
                last_rebalance_portfolio = new_rebalance_portfolio
                last_rebalance_date = current_date
    return return_df


def market_return(start_date, prices, trade_periods):
    return_df = pd.DataFrame(index=prices.index,
                             columns=['Market Index Return', 'Excess Return'])
    return_df.iloc[0] = 1

    last_rebalance_date = prices.index[0]
    for current_date in prices.index[1:]:
        # Fetching index price
        index_close = hs_300.loc[current_date, 'close']
        index_rebalance_close = hs_300.loc[last_rebalance_date, 'close']
        # Calculating cumulative monthly return
        period_return = (index_close / index_rebalance_close) - 1.015
        index_last_cum_return = return_df.at[last_rebalance_date, 'Market Index Return']
        return_df.at[current_date, 'Market Index Return'] = index_last_cum_return * (1 + period_return)

        if current_date in trade_periods:
            last_rebalance_date = current_date

    return return_df


def initializer(start_date, end_date):
    trade_periods = get_rebalance_dates(start_date, end_date)
    filtered_prices_df = price_filtering(start_date, end_date)

    smb_hml_returns = get(filtered_prices_df, 0, trade_periods)
    umd_returns = get(filtered_prices_df, 1, trade_periods)
    index_returns = market_return(start_date, filtered_prices_df, trade_periods)

    results = pd.concat([smb_hml_returns, umd_returns, index_returns], axis=1)

    trade_date_ints = trade_periods
    filtered_results = results[results.index.isin(trade_date_ints)]
    filtered_results = filtered_results.pct_change()

    filtered_results['SMB'] = (
            (filtered_results['sl'] + filtered_results['sm'] + filtered_results['sh']) / 3 -
            (filtered_results['bl'] + filtered_results['bm'] + filtered_results['bh']) / 3
    )

    filtered_results['HML'] = (
            (filtered_results['sh'] + filtered_results['bh']) / 2 -
            (filtered_results['sl'] + filtered_results['bl']) / 2
    )

    filtered_results['UMD'] = filtered_results['UMD_top'] - filtered_results['UMD_btm']
    simplified_results = filtered_results[['SMB', 'HML', 'UMD', 'Market Index Return', 'Portfolio Return']]
    simplified_results.iloc[0] = 0
    return simplified_results


start_date = 20200131
end_date = 20230131
result = initializer(start_date, end_date)


truncated_result = result.iloc[1:]
data = {
    'Date': result.index,
    'Asset_Returns': result['Portfolio Return'],  # Your asset returns data here
    'MKT': result['Market Index Return'],  # Market risk premiums here
    'SMB': result['SMB'],  # SMB factor returns here
    'HML': result['HML'],  # HML factor returns here
    'UMD': result['UMD']  # Momentum factor returns here
}
test_df = pd.DataFrame(data)
test_df.set_index('Date', inplace=True)

# Define the independent variables and add a constant term (intercept)
X = test_df[['MKT', 'SMB', 'HML', 'UMD']]
X = sm.add_constant(X)  # Adds a constant term to the predictor

# Define the dependent variable
y = test_df['Asset_Returns']

# Create a regression model and fit it
model = sm.OLS(y, X).fit()

# Print out the statistics
print(model.summary(), f'Data for regressional analysis ranged from {start_date} to {end_date}.')
