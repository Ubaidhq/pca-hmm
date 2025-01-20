# Methodology

This chapter details the data that was used in the analysis. 

## Data

The data used in developing the model and subsequent backtesting was extracted from the Nasdaq Data Link API.  The Nasdaq API provides access to hundreds of both free and premium datasets from various exchanges.  To minimise the complexity of the model, we only focus on one commodity: CBOT Corn.  This is a premium dataset under the Nasdaq API and is sourced from the Chichago Mercantile Exchange.   

### Data Overview

The data consists of daily historic settle prices for a time period ranging from 01/01/2010 - 26/02/2022.  The pricing unit is Cents per bushel.  Each contract is 5,000 bushels, which is approximately 127 Metric Tons.  The tick size is 1/4 of one cent per pushel or $12.50 per contract.  The futures contracts for CBOT Corn are marked for March, May, July, September and December.  The contracts roll over on the 15th of the contract month, or the preceeding business day, usually a Friday, if the 15th is a non-business day.   

It is worth noting that whilst it is possible to to conduct PCA for _fixed_ contracts, in order to extract the maximum amount of data to ensure a degree of robustness to the results, _rolling_ contracts were instead used. This technicality is best explained via an example.

If historical data for the next 10 monthly _fixed_ futures contracts for a given commodity from todays date (Feb 2022, March 2022, ..., Nov 2022) is sought, then a problem arises as the latter contracts would not have been marked during the initial days of the time period being studied. A solution to this problem is to instead look at _rolling_ contracts.

## Derived Variables
Once the data for the futures contracts has been obtained, it is neccessary to decide on which feature or features are to be fed into the HMM as observations. 

## Augmented-Dicky-Fuller Test

As PCA forms a crucial step in the identification and building of the aforementioned trading strategies, it was necessary to ensure there were no violations of any of the assumptions that are made.

To check for stationarity, the augmented Dickey Fuller (ADF) test is used. An ADF test tests the null hypothesis that a unit root is present in the time series sample, inwhich case the sample would be non-stationary. It is applied to the regression model:

where $α$ is a constant, $β$ the coefficient on a time trend and p the lag order of the autoregressive process. The adfuller function from the statsmodels package in Python was used to carry out ADF tests on all contracts for the commodities studied. The function automatically chooses the number of lags by looking minimize the default AIC criterion. Once ADF statistics have been calculated, they are compared to the critical values at the 5% level. In all cases the null hypothesis is rejected, concluding that our data is stationary.

