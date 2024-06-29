import numpy as np
import pandas as pd
from scipy.stats import f
from itertools import combinations


def iman_davenport_test(n_ranks_arr, confidence_level, arr_order='cols'):
    """
     ---------------------------------------------------------------------------------------------
     Peforms Iman-Davenport Statistic (less conservative Friedman statistic) to test the null 
     hypothesis that the ranks of the different classifiers are statistically significantly with 
     a certain confidence level.
    ----------------------------------------------------------------------------------------------
     Parameters:
    -----------------------------------------------------------------------------------------------
        n_ranks_arr: (np.narray) 
            An array of arrays structured by either classifier's rank for each dataset or by a dataset
            with each classifier's rank.  

        confidence_level: (float)
            the confience level required for the friedman statistic, typical inputs are 90% (0.9) 
            or 95% (0.95)

        arr_order: (str)
            order of observations in n_ranks_arr. 'cols' assumes the internal arrays are ordered by,
            classifier with each element a dataset rank (desired case). 'rows' assumes the internal 
            arrays are ordered by dataset. 
    ----------------------------------------------------------------------------------------------
    Returns:
    ----------------------------------------------------------------------------------------------
        iman_davenport_stat: (float)
            The Iman-Davenport statistic for the null hypothesis that the classifier's different
            ranks significantly vary from each other
            
        critical_f_value: (float)
            The critical value to compare the Iman-Davenport Statistic

        reject_null_hypodict: (bool)
           Outcome whether classifier's ranks significantly vary from each other
    ----------------------------------------------------------------------------------------------  
    """
    # Check input for order of ranks in input array
    arr_order_arwgs = ['cols','rows']
    assert arr_order in arr_order_arwgs, f'arr_order: invalid parameter, options: {arr_order_arwgs}'
    if arr_order == "cols": ## Internal arrays of dataset ranks for each classifier
        col_ordered_arr = n_ranks_arr
    else: ## Internal arrays of classifer ranks for each dataset 
        col_ordered_arr = np.transpose(n_ranks_arr) ## Transform into arrays of dataset ranks for each classifier 
    
    # Calculation of Iman-Davenport Statistic
    chi_square = chi_square_f(col_ordered_arr) ## Calculate Chi_squared_f    
    k = col_ordered_arr.shape[0] ## Number of groups/classifiers
    N = col_ordered_arr.shape[1] ## Number of datasets
    iman_davenport_stat = ((N-1)*chi_square)/(N*(k-1)-chi_square)

    # Calculate Critical F-Value
    dfn = k-1 ## Degrees of freedom for numerator 
    dfd = (k-1)*(N-1) ## Degrees of freedom for denominator 
    critical_f_value = f.ppf(confidence_level, dfn, dfd) 

    # Checking Statical Significance
    reject_null_hypo = iman_davenport_stat > critical_f_value ## Significant if stat > critical_value

    return iman_davenport_stat, critical_f_value, reject_null_hypo


def generate_ranks(row, equal_rank_behav='mean', rank_order:str = 'max'):
    """
    ---------------------------------------------------------------------------------------------
    Generates ranks for a pd.DataFrame row
    ----------------------------------------------------------------------------------------------
    Parameters:
    -----------------------------------------------------------------------------------------------
        row: (np.narray) 
            An array of arrays structured by either classifier's rank for each dataset or by a dataset
            with each classifier's rank.

    equal_rank_behav: (str or int or float) 
        Specifies the behavior if all elements have the same rank. Options are strings that match
        pd.DataFrame aggregate functions (min, mean, median, max) to provide value or a specific 
        numeric rank to assign.
    ----------------------------------------------------------------------------------------------
    Returns:
    ----------------------------------------------------------------------------------------------
        row_ranks: (float)
            The Iman-Davenport statistic for the null hypothesis that the classifier's different
            ranks significantly vary from each other
    ----------------------------------------------------------------------------------------------
    """  
    # Generate ranks from data
    row_ranks = row.rank(method=rank_order, ascending=False).astype(int) 
    #  All Equal Rank behavior 
    ## Aggeration Function Dictionary for behavior when all ranks are the same
    # Generate indices based on the length of 'row'
    indices = np.arange(1,len(row)+1)

    # Aggregation functions on indices
    aggr_funct = {
        'min': np.min(indices),
        'mean': np.mean(indices),
        'median': np.median(indices),
        'max': np.max(indices)
        }
    
    ## Error Message
    valid_input_message = (
    "Invalid equal_rank_behavor input:\n"
    f'Please provide a valid integer, float, or aggr_funct: {list(aggr_funct.keys())}'
    )

    ## Check user input
    assert (isinstance(equal_rank_behav, (int, float)) or equal_rank_behav in aggr_funct.keys()), valid_input_message

    if np.all(row_ranks == row_ranks.iloc[0]): ## Check if all ranks the same
        if equal_rank_behav in list(aggr_funct.keys()): ## Valid Aggragate Function Provided
            row_ranks = aggr_funct[equal_rank_behav] ## Assign Aggragate Function Value 
        else:  ## User input is a specific rank number
            row_ranks = equal_rank_behav ## Assign specific rank number
    return row_ranks


def generate_rank_array_from_dataframe(df, cols_to_rank_lst, equal_rank_behav:str ='mean', rank_order:str = 'max'):
    """
    ---------------------------------------------------------------------------------------------
    Generates a rank array on specified columns from a dataframe that contains the datasets as rows 
    and classifiers as column of values
    ----------------------------------------------------------------------------------------------
    Parameters:
    -----------------------------------------------------------------------------------------------
        df: (pd.Dataframe) 
            A dataframe that contains the datasets as rows and classifiers as column  

        cols_to_rank_ls: (list)
            List of column names in df that will be ranked 

        equal_rank_behav: (str or int or float) 
            Specifies the behavior if all elements have the same rank. Options are strings that match
            pd.DataFrame aggregate functions (min, mean, median, max) to provide value or a specific 
            numeric rank to assign.
    ----------------------------------------------------------------------------------------------
    Returns:
    ----------------------------------------------------------------------------------------------
        rank_array: (narray)
            An array of arrays structured by either classifier's rank for each dataset or by a dataset
            with each classifier's rank.  
    ----------------------------------------------------------------------------------------------
    """  
    # Generate Ranks on specified columns
    df_ranks = df[cols_to_rank_lst].apply(generate_ranks, args=(equal_rank_behav,rank_order,), axis=1) 
    # Create array 
    rank_columns =[]
    for i,col in enumerate(cols_to_rank_lst): #F changes DF (feature or bug?)
        df[col + '_rank'] = df_ranks.apply(lambda x: x.iloc[i], axis=1) 
        rank_columns.append(col + '_rank')
    rank_array = df[rank_columns].values.T # Convert to narray

    return rank_array

def chi_square_f(n_ranks_arr):
    """
    ---------------------------------------------------------------------------------------------
    Computation of Chi Sqaured value to used for calculation of Iman-Davenport Statistic
    ----------------------------------------------------------------------------------------------
    Parameters:
    -----------------------------------------------------------------------------------------------
        n_ranks_arr: (np.narray) 
            An array of arrays structured by either classifier's rank for each dataset or by a dataset
            with each classifier's rank.  
    ----------------------------------------------------------------------------------------------
    Returns:
    ----------------------------------------------------------------------------------------------
    chi: (float)
        Chi Statistic 
    ----------------------------------------------------------------------------------------------
    """  
    # Calculation of CHI
    k = n_ranks_arr.shape[0] ## Number of groups/classifiers
    N = n_ranks_arr.shape[1] ## Number of datasets or runs for each classifier/group
    means = np.mean(n_ranks_arr,axis=1) ## Mean rank for each classifier/group
    sum_means_sqr = np.power(means,2).sum() # Sum of means^2
    chi = (12*N)/(k*(k+1))*(sum_means_sqr-((k*np.power(k+1,2)/4))) # Computation of CHI
    return chi

def nemenyi_test(n_ranks_arr,confidence_level, clf_names):
    """
    ---------------------------------------------------------------------------------------------
    Peforms Nemeny Test on each combination of classifiers to test the null hypothesis that there
    is a statisically significant (90% or 95% confidence interval) rank difference between the classifiers. 
    ----------------------------------------------------------------------------------------------
    Parameters:
    -----------------------------------------------------------------------------------------------
        n_ranks_arr: (np.narray) 
            An array of arrays structured by either classifier's rank for each dataset 
    
        confidence_level: (float)
            the confience level required for the nemenyi test, allowed inputs are 90% (0.9) 
            or 95% (0.95)

        clf_names: (list)
            Name of the different classifiers/groups
    ----------------------------------------------------------------------------------------------
    Returns:
    ----------------------------------------------------------------------------------------------
        results: list(clf_pair, mean_diff, crit_diff,reject_null_hypo)
            List of tuples where each tuple is a summary of the nemeny test for a classifier pair

                clf_pair: (str,str) 
                    Identifies which classifiers are paired

                mean_diff: (float) 
                    Difference between the classifier's mean 

                crit_diff: (float) 
                    Threshold for statistical significance                   

                reject_null_hypodict: (bool)
                    Outcome whether classifier's ranks significantly vary from each other
    ---------------------------------------------------------------------------------------------- 
    """ 
    # Checking if inputed Confidence Interval is in tabular data
    confidence_arwgs = (0.90, 0.95) 
    assert confidence_level in confidence_arwgs, f'arr_order: invalid parameter, options: {confidence_arwgs}'
    demsar_dic = {2: {'0.05': 1.960, '0.10': 1.645},
                  3: {'0.05': 2.343, '0.10': 2.052},
                  4: {'0.05': 2.567, '0.10': 2.291},
                  5: {'0.05': 2.728, '0.10': 2.459},
                  6: {'0.05': 2.850, '0.10': 2.589},
                  7: {'0.05': 2.949, '0.10': 2.693},
                  8: {'0.05': 3.031, '0.10': 2.780},
                  9: {'0.05': 3.102, '0.10': 2.855},
                  10: {'0.05': 3.164, '0.10': 2.920}}

    # Calculation of Critical Difference (Null Hypothesis Threshold)
    num_of_clf = n_ranks_arr.shape[0]
    alpha = str(np.round(1-confidence_level,2))
    N = n_ranks_arr.shape[1]
    k = demsar_dic[num_of_clf][alpha]
    crit_diff = k * np.power((num_of_clf * (num_of_clf +1))/(6*N),0.5)

    # Computing Mean Rank for each classifier
    means = np.mean(n_ranks_arr,axis=1)
    means_dic = {clf: mean for clf,mean in zip(clf_names,means)}

    # Computing Mean DIfference, comparing against CD and evaulating Null hypotheses
    results = []
    for clf_pair in combinations(clf_names,2):
        mean_diff = np.abs(means_dic[clf_pair[0]] - means_dic[clf_pair[1]])
        reject_null_hypo = mean_diff > crit_diff
        results.append((clf_pair,mean_diff,crit_diff,reject_null_hypo))
    
    
    return results 