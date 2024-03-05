import pandas as pd
import numpy as np
from scipy.stats import t
from scipy.linalg import svd
from scipy.spatial.distance import cdist
from sklearn.cluster import OPTICS
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

# Proposed pairs selection framework: Section A
class PCA:
    """
    Principal Component Analysis (PCA) class that performs dimensionality reduction on a dataset by computing the
    principal components of the data and projecting it onto a lower-dimensional space.

    Parameters:
    -----------
    n_components : int
        Number of principal components to compute.

    Attributes:
    -----------
    explained_variance_ratio_ : ndarray, shape (n_components,)
        Proportion of variance explained by each principal component.

    components_ : ndarray, shape (n_features, n_components)
        Principal components of the data.

    transformed_data_ : pandas DataFrame, shape (n_samples, n_components)
        Transformed data after projection onto the lower-dimensional space.
    """

    def __init__(self, n_components):
        self.n_components = n_components
        self.explained_variance_ratio_ = None
        self.components_ = None
        self.transformed_data_ = None

    def fit_transform(self, X):
        """
        Fits the PCA model to the input data and transforms it into the lower-dimensional space.

        Parameters:
        -----------
        X : pandas DataFrame, shape (n_samples, n_features)
            Input data to transform.

        Returns:
        --------
        transformed_data : pandas DataFrame, shape (n_samples, n_components)
            Transformed data after projection onto the lower-dimensional space.
        """

        # Standardize the data
        X_std = (X - X.mean()) / X.std()

        # Compute the SVD of the standardized data
        '''
        We then compute the SVD of the standardized data using the scipy.linalg.svd function.
        The full_matrices=False argument is used to indicate that we want to compute the reduced SVD,
        which returns only the top min(n_samples, n_features) singular vectors, since any remaining singular
        vectors would not contribute to the top principal components.
        '''
        _, s, Vt = svd(X_std, full_matrices=False)
        V = Vt.T

        # Select the top n_components columns of V to obtain the selected components matrix
        selected_components = V[:, :self.n_components]

        # Compute the transformed data by projecting X_std onto the selected components
        transformed_data = np.dot(X_std, selected_components)

        # Compute the explained variance ratios
        total_var = np.sum(np.var(X_std, axis=0))
        explained_var = np.var(transformed_data, axis=0)
        explained_var_ratio = explained_var / total_var

        # Store the results as attributes of the PCA object
        self.explained_variance_ratio_ = explained_var_ratio
        self.components_ = selected_components
        self.transformed_data_ = pd.DataFrame(transformed_data)

        return self.transformed_data_

# Proposed pairs selection framework: Section B
class Optics:
    """
    A class to fit a clustering model using the OPTICS algorithm. OPTICS 
    (Ordering Points To Identify the Clustering Structure), closely
    related to DBSCAN, finds core sample of high density and expands clusters
    from them. Unlike DBSCAN, keeps cluster hierarchy for a variable
    neighborhood radius. Better suited for usage on large datasets than the
    current sklearn implementation of DBSCAN.

    Parameters:
    -----------
    stocksNames : list
        A list of the names of the stocks being clustered.
    min_samples :  min_samples : int > 1 or float between 0 and 1, default=5
        The number of samples in a neighborhood for a point to be considered as a core point.
    max_eps : float, default=np.inf
        The maximum distance between two samples for one to be considered as
        in the neighborhood of the other.
    xi : float between 0 and 1, default=0.05
        Determines the minimum steepness on the reachability plot that
        constitutes a cluster boundary. Used only when ``cluster_method='xi'``.
    cluster_method : str, default='xi'
        The extraction method used to extract clusters using the calculated
        reachability and ordering. Possible values are "xi" and "dbscan".
    dist_metric : str, default='euclidean'
        Metric to use for distance computation. Any metric from scikit-learn
        or scipy.spatial.distance can be used.
    n_jobs : int, default=None
        The number of parallel jobs to run.

    Attributes:
    -----------
    clf : sklearn.cluster.OPTICS
        The trained OPTICS clustering model.
    clsSeries : pandas.Series
        A pandas series containing the cluster labels for each data point, with the stock names as the index.
    nClusters : int
        The number of clusters in the data.
    """

    def __init__(self, stocksNames, min_samples=5, max_eps=np.inf, xi=0.05, cluster_method='xi', dist_metric='euclidean', n_jobs=None):
        self.stocksNames = stocksNames
        self.min_samples = min_samples
        self.max_eps = max_eps
        self.xi = xi
        self.cluster_method = cluster_method
        self.dist_metric = dist_metric
        self.n_jobs = n_jobs
        self.clf = None
        self.clsSeries = None
        self.nClusters = None
        
    
    def fit(self,X):
        """
        Fits the OPTICS clustering model to the input data X, and stores the trained model, cluster labels, and number
        of clusters as attributes of the Optics class.

        Parameters:
        -----------
        X : pandas.DataFrame, shape (n_samples, n_features)
            The input data to cluster.

        Returns:
        --------
        clsSeries: pandas.Series, shape (n_samples, cluster labels)
            A pandas series containing the cluster labels for each data point, with the stock names as the index.
        """        
        # Create an OPTICS object with the specified parameters and fit it to the input data X.
        clf = OPTICS(min_samples=self.min_samples,max_eps=self.max_eps, xi=self.xi, cluster_method=self.cluster_method, metric=self.dist_metric, n_jobs=self.n_jobs)
        clf.fit(X)

        # Extract the cluster labels for each data point.
        labels = clf.labels_
        nClusters = len(set(labels)) - (1 if -1 in labels else 0)

        # Create a pandas series with the cluster labels for each stock, with the stock names as the index.
        stocksClsAll = pd.Series(index=self.stocksNames, data=labels.flatten())
        clsSeries = stocksClsAll[stocksClsAll != -1]

        # Store the trained OPTICS model, the cluster labels, and the number of clusters as attributes of the Optics class.
        self.clf = clf
        self.clsSeries = clsSeries
        self.nClusters = nClusters
        
        # Print how many cluster the algorithm discovered
        counts = clsSeries.value_counts()
        print("Clusters discovered: %d" % nClusters)

        # Cluster labels
        return clsSeries

# Proposed pairs selection framework: Section C
def checkCointegration(stockX,stockY,alpha=0.05):
    """
    Performs the Engle-Granger two-step cointegration test for two I(1) time series in both directions.

    Parameters:
    -----------
    stockX : pandas.Series
        The first time series to test for cointegration.
    stockY : pandas.Series
        The second time series to test for cointegration.
    alpha : float, optional
        The significance level for the test. Default is 0.05.

    Returns:
    --------
    dict or bool
        A dictionary containing the cointegrated formula, residuals, t-statistic, and regression beta if the two time series are 
        cointegrated in both direction, False otherwise.
    """
    
    # Step 0: Get names for each series
    x_name = stockX.name
    y_name = stockY.name

    # Step 1: Verify that both series are I(1)
    x_is_stationary = adfuller(stockX)[1] > alpha
    y_is_stationary = adfuller(stockY)[1] > alpha

    if not (x_is_stationary and y_is_stationary):
        # If either time series is not stationary, we cannot perform the cointegration test.
        return {'coint': False,'y':np.nan, 'x':np.nan, 'error':np.nan, 't-stat':np.nan, 'beta':np.nan}
    
    # Step 2: Check for cointegration using the Engle-Granger test in both directions.
    series = pd.concat([stockX.shift(1), stockY.shift(1)], axis=1).dropna().values
    
    # Y ~ X
    model1 = sm.OLS(endog=stockY.values, exog=sm.add_constant(stockX.values)).fit()
    x_beta = model1.params[1]
    y_resid = model1.resid
    y_t_stat_1 = sm.tsa.stattools.adfuller(y_resid)
    y_p_value = y_t_stat_1[1]
    y_t_stat_1 = y_t_stat_1[0]
    y_is_cointegrated_1 = y_p_value < alpha

    # X ~ Y
    model2 = sm.OLS(endog=stockX.values, exog=sm.add_constant(stockY.values)).fit()
    y_beta = model2.params[1]    
    x_resid = model2.resid
    x_t_stat_1 = sm.tsa.stattools.adfuller(x_resid)
    x_p_value = x_t_stat_1[1]    
    x_t_stat_1 = x_t_stat_1[0]
    x_is_cointegrated_1 = x_p_value < alpha

    # Step 3: Decision
    if y_is_cointegrated_1 and x_is_cointegrated_1:
        # Both directions show cointegration.
        if abs(y_t_stat_1) < abs(x_t_stat_1):
            # Use Y ~ X as the cointegrated formula.
            return {'coint': True,'y':stockY, 'x':stockX, 'error':y_resid, 't-stat':y_t_stat_1, 'beta':x_beta}
        else:
            # Use X ~ Y as the cointegrated formula.
            return {'coint': True, 'y':stockX, 'x':stockY, 'error':x_resid, 't-stat':x_t_stat_1, 'beta':y_beta}
    else:
        # No cointegration was found.
        return {'coint': False,'y':np.nan, 'x':np.nan, 'error':np.nan, 't-stat':np.nan, 'beta':np.nan}

def hurst_exponent(residual):
    """
    Estimates the Hurst exponent of a time series based on its residual.
    Source: https://www.quantstart.com/articles/Basics-of-Statistical-Mean-Reversion-Testing

    Parameters:
    -----------
    residual : pandas.Series
        The residual of the cointegration test.

    Returns:
    --------
    float
        The estimated Hurst exponent.
    """        
    # Create the range of lag values
    lags = range(2, 100)

    # Calculate the array of the variances of the lagged differences
    # Here it calculates the variances, but why it uses
    # standard deviation and then make a root of it?
    tau = [np.sqrt(np.std(np.subtract(residual[lag:], residual[:-lag]))) for lag in lags]

    # Use a linear fit to estimate the Hurst Exponent
    poly = np.polyfit(np.log(lags), np.log(tau), 1)

    # Return the Hurst exponent from the polyfit output
    hurst_exponent = poly[0] * 2.0
    return hurst_exponent


def calculate_half_life(residuals):
    """
    This function calculates the half life parameter of a mean reversion series. The half-life of 
    mean-reversion is an indicator of how long it takes for a time series to mean revert

    Parameters:
    -----------
    residual : pandas.Series
        The residual of the cointegration test.

    Returns:
    --------
    float
        The estimated half-life.
    """
    z_lag = np.roll(residuals, 1)
    z_lag[0] = 0
    z_ret = residuals - z_lag
    z_ret[0] = 0

    # adds intercept terms to X variable for regression
    z_lag2 = np.vstack([np.ones(len(z_lag)), z_lag]).T
    beta = np.dot(np.linalg.inv(np.dot(z_lag2[1:].T, z_lag2[1:])), np.dot(z_lag2[1:].T, z_ret[1:]))
    halflife = -np.log(2) / beta[1]
    halflife = int(round(halflife))

    return halflife

def zero_crossings(residuals,index):
    """
    This function calculates the number of times a spread crosses its mean on average per year

    Parameters:
    -----------
    residual : pandas.Series
        The residual of the cointegration test.

    Returns:
    --------
    float
        The number of times
    """
    residuals = pd.DataFrame(residuals, index=index)
    sign = np.sign(residuals)
    grouped = sign.groupby(pd.Grouper(freq='Y'))
    crossings = grouped.apply(lambda x: (x[:-1].reset_index(drop=False) != x[1:].reset_index(drop=False)).sum())
    crossings_mean = crossings / 2
    zero_crossings = crossings_mean.mean()
    zero_crossings = zero_crossings.to_list()[1]

    return zero_crossings


class getPairs:
    """
    A class to get pairs of cointegrated stocks based on specified criteria.

    Attributes
    ----------
    alpha : float, optional
        The significance level for the cointegration test (default is 0.05).
    hurstThresh : float, optional
        The threshold value for the Hurst exponent (default is 0.5).
    minhLife : int, optional
        The minimum value for the half-life (default is 22).
    maxhLife : int, optional
        The maximum value for the half-life (default is 132).
    zeroCrosses : int, optional
        The minimum number of zero crossings for the spread (default is 0).
    universe : dict, optional
        A dictionary containing the number of pairs, cointegrated pairs, pairs with
        Hurst exponent below hurstThresh, pairs with half-life between minhLife and
        maxhLife, and pairs with more than zeroCrosses number of zero crossings.
    pairs : list, optional
        A list containing dictionaries of cointegrated pairs and their statistics.

    Methods
    -------
    eval(clusters, prices):
        Evaluates if the potential pairs are cointegrated based on specified criteria.

    """

    def __init__(self, alpha=0.05, hurstThresh=0.5, minHLife=22, maxHLife=132, zeroCrosses=0):
        self.alpha = alpha
        self.hurstThresh = hurstThresh
        self.minHLife = minHLife
        self.maxHLife = maxHLife
        self.zeroCrosses = zeroCrosses
        self.universe = None
        self.pairs = None

    # Step 1: Evaluate if the potential pairs are cointegrated
    def eval(self, clusters, prices):
        """
        Multi-step evaluation based on different criteria like cointegration, hurst exponent, half-life and zero mean crosses.

        Parameters
        ----------
        clusters : pandas.DataFrame
            A DataFrame containing the cluster IDs of the stocks.
        prices : pandas.DataFrame
            A DataFrame containing the prices of the stocks.

        Returns
        -------
        pairsList : list
            A list of dictionaries containing successful pairs and their statistics.

        """        
        universe = {'pairs':0,'cointegrated':0,'hurst':0,'half-life':0,'zero-cross':0}
        
        # Check for cointegration
        pairsList = []
        pairs = []
        clusterIDs = list(np.unique(clusters.values))     
        for cluster in clusterIDs:
            symbols = clusters.loc[clusters == cluster,].index.to_list()
            for i in range(len(symbols)):
                for j in range(i+1,len(symbols)):
                    pair = sorted([symbols[i], symbols[j]])
                    if pair not in pairs:
                        X = prices[symbols[i]]
                        Y = prices[symbols[j]]
                        output = checkCointegration(X, Y, alpha = self.alpha)
                        pair_name = f"{symbols[i]}_{symbols[j]}"
                        output['pair'] = pair_name
                        pairsList.append(output)
                        pairs.append(pair)

        universe['pairs'] = len(pairs)
        tempList = [d for d in pairsList if d['coint'] == True]
        if len(tempList) == 0:
            raise ValueError("No cointegrated pairs found.")
                
        pairsList = tempList
        universe['cointegrated'] = len(pairsList)

        # Check for hurst exponent
        for idx in range(len(pairsList)):
            pairsList[idx]['hurst'] = hurst_exponent(pairsList[idx]['error'])        
        tempList = [d for d in pairsList if d['hurst'] < self.hurstThresh]
        if len(tempList) == 0:
            raise ValueError("No pairs found with Hurst exponent below threshold.")
        pairsList = tempList
        universe['hurst'] = len(pairsList)

        # Check for half-life
        for idx in range(len(pairsList)):
            pairsList[idx]['hlife'] = calculate_half_life(pairsList[idx]['error'])        
        tempList = [d for d in pairsList if d['hlife'] > self.minHLife] 
        tempList = [d for d in tempList if d['hlife'] < self.maxHLife]
        if len(tempList) == 0:
            raise ValueError("No pairs found with half-life within specified range.")
        pairsList = tempList
        universe['half-life'] = len(pairsList) 

        # Check for zero-crosses
        for idx in range(len(pairsList)):
            pairsList[idx]['zero_cross'] = zero_crossings(pairsList[idx]['error'], pairsList[idx]['y'].index)
        tempList = [d for d in pairsList if d['zero_cross'] > self.zeroCrosses]
        if len(tempList) == 0:
            raise ValueError("No pairs found with more than specified zero-crossings.")
        pairsList = tempList
        universe['zero-cross'] = len(pairsList)

        # Store result
        self.universe = universe
        self.pairs = pairsList

        # Return pairs
        return pairsList