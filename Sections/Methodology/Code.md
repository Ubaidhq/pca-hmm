### HMM - Regime Switch
- Need to check if the data is stationary
- We can create derived variables to feed into our HMM because we don't want it to essentially be learning
- What variables do we want to use?
	- The slope of the curve, ie the spread between short term and long term contracts
	- a proxy for the volatility
		- This can be particularly effective for regime classification because of volatility clustering. 
	- the mean. 

### Model Configuration

There are three primary parameters for thhe model that we can define: 
 1. The number of hidden states,
 2. The covariance type
 3. Threshold for maximum number of iterations to perform for the EM algorithm.

An important consideration for determining the parameters for the HMM is the bias-variance tradeoff.  Both the number of hidden states and the threshold for EM can influence the fit of the model.  

We probably want to use a full covariance matrix?  

## Derived Variables 
- How do we calculate the derived volatility? 
	- We are using the MSE. 
	- We have an x day window which we use to calculate the mean at any time.  
	- We can then create an array of length x with the mean which we can then use to work out the mean squared error at each point. 
		- The issue here is we don't just have one observation on any given date, we have 10.  
		- Therefore we are taking an x day mean of means of the 10 contracts.  
		`df_temperature.average_temperature.rolling(10, min_periods = 1).mean`
	* We have a column of the moving average means.
* We want to calculate a dervied slope.  
	* This can be done by calculating the difference between the 10th contract ahead and the first contract ahead.  

### Mutiple Variables Implementation
- We can do just with volatillity, level or slope - 3 runs. 
- We can then do without any regimes at all as our most basic version.
- We can then use our multiple observation method if we have time? 


### Commodities
- It will be interesteing to do apply the strategy to different agricultural commodities almost as a portfolio of strategies and see how this performs.
- We can then do the summary results and plots for all the commodities in onne big plot.  