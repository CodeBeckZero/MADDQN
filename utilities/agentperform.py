import matplotlib.pyplot as plt
import numpy as np
import quantstats as qs
import pandas as pd
import logging
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker

def agent_stock_performance(stock_price_ts: np.ndarray, trade_ts: np.ndarray,
                            stock_name: str, agent_name:str, 
                            display_graph: bool = False, 
                            save_graphic: bool = False,
                            path_file = None):
    """
    Analyzes the trading performance of an agent on a stock dataset.

    Parameters:
    - stock_price_ts (np.ndarray): 1-D array with stock's price at each timestep.
    - trade_ts (np.ndarray): 1-D array with agent's action at each time step. Actions defined as [-1, 0, 1] for [Sell, Hold, Buy] respectively.
    - stock_name (str): Name of the stock, used for labeling plots.
    - agent_name (str): Name of the agent, used for labeling plots.
    - display_graph (bool, optional): Whether to display the generated plot. Defaults to False.
    - save_graphic (bool, optional): Whether to save the plot as an image. Defaults to False.
    - path_file (str, optional): Path to save the generated plot if save_graphic is True. Defaults to None.

    Returns:
    - dict: Dictionary containing agent's trade performance metrics:
        - 'stock': Name of the stock.
        - 'agent_name': Name of the agent.
        - 'n_trades': Number of trades.
        - 'n_wins': Number of winning trades.
        - 'n_losses': Number of losing trades.
        - 'win_percentage': Trade win percentage.
        - 'cumulative_return': Cumulative return.
        - 'sortino': Sortino ratio.
        - 'max_drawdown': Maximum drawdown percentage.
        - 'sharpe': Sharpe ratio.
        - 'trade_dur_avg': Average duration of trades.
        - 'trade_dur_min': Minimum duration of trades.
        - 'trade_dur_max': Maximum duration of trades.
        - 'buy_hold': Buy and Hold return.
    """
    
        
    # Finding index and stock price of Buy Action
    buy_price_idx = np.where(trade_ts == 'B')[0]
    logging.info(buy_price_idx)
    buy_price_idx = buy_price_idx.astype(int)
    buy_price = stock_price_ts[buy_price_idx]
    

    # Finding index and stock price of Sell Action
    sell_price_idx = np.where(trade_ts == 'S')[0]
    sell_price_idx = sell_price_idx.astype(int)
    sell_price = stock_price_ts[sell_price_idx]
    logging.info(sell_price_idx)
    # Error Checking from enviroment
    assert len(buy_price_idx) == len(sell_price_idx),\
          "Trade action input didn't produce equal buy and sell actions"
    

    for i in range(len(buy_price_idx)):
        assert buy_price_idx[i] < sell_price_idx[i],\
            f"Assertion failed at index {i}: {buy_price_idx[i]} is not smaller than {sell_price_idx[i]}"
    

    if len(buy_price_idx) != 0:

        # Calculating Win, Loss, Total Trades
        trade_wins = np.sum(stock_price_ts[buy_price_idx] < stock_price_ts[sell_price_idx])
        trade_loss = np.sum(stock_price_ts[buy_price_idx] > stock_price_ts[sell_price_idx])
        trade_total = int((len(buy_price_idx) + len(sell_price_idx))/2) #Function assumes trade_ts has proper buy-sell patterns 
        win_precentage = trade_wins/trade_total*100

        # Calculating Average Length of Trade
        trade_lengths = np.array(sell_price_idx - buy_price_idx)
        trade_length_min = np.min(trade_lengths)
        trade_length_avg = np.mean(trade_lengths)      
        trade_length_max = np.max(trade_lengths)
        
        # Caculating Financial Performance

        returns_list = []

        for buy_index, sell_index in zip(buy_price_idx,sell_price_idx):
            trade_return = (stock_price_ts[sell_index] - stock_price_ts[buy_index])\
                /stock_price_ts[buy_index]

            
            returns_list.append(trade_return)
        


        returns = pd.Series(returns_list)
        sharpe_ratio = qs.stats.sharpe(returns)
        mdd = qs.stats.max_drawdown(returns)*100
        cumulative_return = returns.add(1).prod()
        sortino_ratio = qs.stats.sortino(returns)

        bad_values = [np.inf]

        if sharpe_ratio in bad_values:
            sharpe_ratio = 0
        
        if sortino_ratio in bad_values:
            sortino_ratio = 0

        if np.isnan(sharpe_ratio):
            sharpe_ratio = 0

        if np.isnan(sortino_ratio):
            sortino_ratio = 0

    else: 
        trade_wins = 0
        trade_loss = 0
        trade_total = 0 
        win_precentage = 0
        trade_lengths = 0
        trade_length_min = 0
        trade_length_avg = 0   
        trade_length_max = 0
        mdd = 0
        sharpe_ratio = 0
        cumulative_return = 0
        sortino_ratio = 0 

    # Buy and Hold
    bh_return = stock_price_ts[-1] / stock_price_ts[0]
 
    results = {'stock': stock_name,
            'agent_name': agent_name,
            "n_trades": trade_total, 
            "n_wins": trade_wins, 
            "n_losses": trade_loss, 
            "win_percentage":win_precentage, 
            "cumulative_return":cumulative_return,
            "sortino": sortino_ratio,
            'max_drawdown': mdd,
            'sharpe': sharpe_ratio,
            'trade_dur_avg': trade_length_avg,
            'trade_dur_min': trade_length_min,
            'trade_dur_max': trade_length_max,
            'buy_hold': bh_return}


    if display_graph is False and save_graphic is False:    
        return results
    
    # Ploting Stock Price and locations of Buy and Sell Actions
    fig, ax = plt.subplots()
    ax.plot(stock_price_ts, color='grey')
    ax.scatter(buy_price_idx,buy_price,color='g',marker="^")
    ax.scatter(sell_price_idx,sell_price,color='r',marker="v")


    hl_start = np.where(trade_ts == 'B')[0]
    hl_end = np.where(trade_ts == 'S')[0]

    for start, end in zip(hl_start, hl_end):
        color = 'grey' if stock_price_ts[start] == stock_price_ts[end] \
            else ('g' if stock_price_ts[start] < stock_price_ts[end] else 'r')
        plt.axvspan(start, end, color=color, alpha=0.15)

    plt.title(f'{agent_name}: {stock_name} Trade Performance')
    plt.ylabel(f'{stock_name} Price')
    plt.xlabel('Time Step$_{Test\ Range}$ ')
    
    # Positioning of Performance Number Text
    # Manual identification of stocks that require test located else where
    
    top_middle_align = ['COKE','BRK']
    
    
    if stock_name in top_middle_align:

    
        plotbox_x = np.median(range(len(trade_ts)))
        plotbox_y = ((np.max(stock_price_ts) - np.min(stock_price_ts))*(2/3)) \
            + np.min(stock_price_ts) 
    
    else:
        plotbox_x = np.min(range(len(trade_ts)))
        plotbox_y = np.min(stock_price_ts)
        
    texbox_content = (f"Trades\n"
        f'(# : W : L : W%):\n'
        f'{trade_total} : {trade_wins} : {trade_loss} : {win_precentage:.1f}\n '
        f'\nTrade Duration\n'
        f'(min : avg : max):\n'
        f'{trade_length_min:.2f} : {trade_length_avg:.2f} : {trade_length_max:.2f}\n'           
        
        f'\nFinancials\n'
        f'(CR : SP : SOR : MDD%):\n'
        f'{cumulative_return:.2f} : {sharpe_ratio:.2f} : {sortino_ratio:.2f} : {mdd:.1f}\n'
        
        f'\nB&H: {bh_return:.2f}'                
    )
    ax.text(plotbox_x,
            plotbox_y,
            texbox_content, 
            bbox=dict(facecolor='yellow', alpha=0.35),
            ha='left',
            va='bottom',
            fontsize=8)      
    if save_graphic:
        assert path_file is not None, "No path/filename provided"

        fig.savefig(path_file)

        if not display_graph:
            plt.close()
            return results
    
    plt.show()
    plt.close()
       
    return results

def update_trade_df(df, trade_seq:list):
    """
    Update a DataFrame based on a sequence of trade actions.

    Parameters:
    - df (pd.DataFrame): The original DataFrame to be updated.
    - trade_seq (list): List of trade actions, where each element is 'B' (Buy) or 'S' (Sell).

    Returns:
    pd.DataFrame: An updated DataFrame with incremented trade actions and position counts.

    Raises:
    AssertionError: If any trade action column is missing in the DataFrame.
    """

    assert all(column in df.columns for column in set(trade_seq)), "Columns are missing in DataFrame"
    assert df.shape[0] == len(trade_seq), f'Trade action list len{len(trade_seq)}!= DF rows{df.shape[0]}' 
    
    updated_df = df.copy()
    pos_list = position(trade_seq)
    for idx,action in enumerate(trade_seq):
        
        updated_df[action].iloc[idx] += 1
        updated_df['Position_Count'].iloc[idx] += pos_list[idx]
    
    return updated_df


def position(trade_seq:list):
    """
    Calculate positions based on a sequence of trade actions.

    Parameters:
    - trade_seq (list): List of trade actions, where each element is 'B' (Buy) or 'S' (Sell).

    Returns:
    np.ndarray: An array indicating positions, where 1 represents holding a position, and 0 represents no position.

    Raises:
    AssertionError: If the number of buy and sell actions in the trade sequence is not equal.
    """
    trade_arr = np.array(trade_seq)
    buy_idx = np.where(trade_arr == 'B')[0]
    sell_idx = np.where(trade_arr == 'S')[0]

    assert len(buy_idx) == len(sell_idx),\
          "Trade action input didn't produce equal buy and sell actions"

    pos_list = np.zeros(len(trade_seq),dtype=int)

    for buy, sell in list(zip(buy_idx,sell_idx)):
        pos_list[buy:sell] = 1
    
    return pos_list

def aggregate_trade_performance(df_trade_action:pd.DataFrame,
                            stock_name: str, 
                            agent_name:str,
                            num_tests: int, 
                            display_graph: bool = False, 
                            save_graphic: bool = False,
                            path_file = None):
    """
    Aggregate and visualize trade performance across multiple tests.

    Parameters:
    - df_trade_action (pd.DataFrame): DataFrame containing trade actions data.
    - stock_name (str): Name of the stock.
    - agent_name (str): Name of the trading agent.
    - num_tests (int): Number of tests.
    - display_graph (bool, optional): Whether to display the graph. Default is False.
    - save_graphic (bool, optional): Whether to save the graph. Default is False.
    - path_file (str, optional): Path and filename to save the graph.

    Returns:
    None

    Raises:
    AssertionError: If `save_graphic` is True but no path/filename is provided.
    """

    if display_graph is False and save_graphic is False:    
        return 

    test_idx = [0,df_trade_action.shape[0]]
    df_trade_action['S'] = 0-df_trade_action['S']

    # Create a figure and a 2x1 grid of subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})

    # Line graph for 'close' column
    ax1.plot(df_trade_action['close'], label='Close Price', color='black')
    ax1.set_xticks([])
    ax1.set_xticklabels([])
    ax1.set_xlim(test_idx[0],test_idx[-1])
    ax1.set_ylabel(f'{stock_name} Price')
    ax1.tick_params('y', colors='black')
    ax1.set_title(f'{agent_name}: {stock_name} Trade Results Across {num_tests} Tests')


    # Create a second y-axis for stacked bar graph
    ax1a = ax1.twinx()
    buy_values = df_trade_action['B'].values
    sell_values = df_trade_action['S'].values
    plot_min_max =  buy_values.max() * 3
    ax1a.bar(df_trade_action.index, buy_values, color='green', alpha=0.7, width=1)
    ax1a.bar(df_trade_action.index, sell_values, color='red', alpha=0.7, width=1)
    ax1a.set_xticks([])
    ax1a.set_xticklabels([])

    # Set y-axis label with different colors
    ybox1 = TextArea("Sell ", textprops=dict(color="red", size=10,rotation=90,ha='left',va='bottom'))
    ybox2 = TextArea(" & ",     textprops=dict(color="black", size=10,rotation=90,ha='left',va='bottom'))
    ybox3 = TextArea("Buy", textprops=dict(color="green", size=10,rotation=90,ha='left',va='bottom'))
    ybox4 = TextArea("Actions", textprops=dict(color="black", size=10,rotation=90,ha='left',va='bottom'))


    ybox_1 = VPacker(children=[ybox3, ybox2, ybox1],align="center", pad=0, sep=5)

    anchored_ybox1 = AnchoredOffsetbox(loc=6, child=ybox_1, pad=0., frameon=False, bbox_to_anchor=(1.03,0.495), 
                                    bbox_transform=ax1a.transAxes, borderpad=0.)
    ax1a.add_artist(anchored_ybox1)

    ybox4 = TextArea("Actions", textprops=dict(color="black", size=10,rotation=90,ha='left',va='bottom'))
    ybox_2 = VPacker(children=[ybox4],align="center", pad=0, sep=5)
    anchored_ybox2 = AnchoredOffsetbox(loc=6, child=ybox_2, pad=0., frameon=False, bbox_to_anchor=(1.05,0.495), 
                                    bbox_transform=ax1a.transAxes, borderpad=0.)

    ax1a.add_artist(anchored_ybox2)

    ax1a.set_ylim(-plot_min_max,plot_min_max)

    custom_labels = [int(float(str(tick).replace('-', ''))) for tick in ax1a.get_yticks()]
   

    for label, tick in list(zip(ax1a.get_yticklabels(), ax1a.get_yticks())):
        formatted_tick = float(str(tick).replace('−', ''))
  
        if int(formatted_tick) > 0:
            label.set_color('green')
        elif int(formatted_tick) < 0:
            label.set_text(f'coke')
            label.set_color('red')
        else:
            label.set_color('black')

        # Modify the label to display the absolute value

    ax1a.set_yticklabels(custom_labels)


    # Line graph for position count
    ax2.plot(df_trade_action['Position_Count']/num_tests, label='Position Count', color='orange')
    ax2.set_xlabel('Time Step$_{Test\ Range}$ ')
    ax2.set_ylabel('Position Probability')
    ax2.set_xlim(test_idx[0],test_idx[-1])

    plt.subplots_adjust(hspace=0.0)
    plt.tight_layout()
    if save_graphic:
        assert path_file is not None, "No path/filename provided"

        fig.savefig(path_file)

        if not display_graph:
            plt.close()
            return        
    
    plt.show()
    plt.close()
    return

def graph_input(stock_prices,
                stock_name: str,
                display_graph: bool = False, 
                save_graphic: bool = False,
                path_file = None):
    
    if display_graph is False and save_graphic is False:    
        return 

    fig, ax = plt.subplots()
    ax.plot(stock_prices, color='grey')
    ax.set_xlim(0,len(stock_prices))
    plt.title(f'{stock_name} Input Data')
    plt.ylabel(f'Price')
    plt.xlabel('Time Steps')
    print('here')
    if save_graphic:
        assert path_file is not None, "No path/filename provided"
        fig.savefig(path_file)

        if not display_graph:
            plt.close()
            return        

    plt.show()
    plt.close()
    return
