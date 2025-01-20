What plots are we using, how are we analysing them and why are we using them? 

**1. Contour plots**
	1. Look at the different contours to see how the data is distributed among the different hidden states
	2. This will help us identify what an appropriate number of states is.
	3. Maybe individual plots of the distributiosn here would be good also
2. **Plotting the training time series broken down into regimes**
	1. As above we want a general picture of what each state is trying to capture
	2. Given the input, it would be most easy to identify states with high/low level and high/low volatility.  
3. **ACF plot**
	1. Low correlation scores
	2. Can be used to show correlation (or lack of) at different lags. 
	3. We do this to show the lack of correlation between log returns and volatility as opposed to to level and volatiltiy.  
4. **Cumulative Profit History**
	1. Benchmark strategy
	2. For all holding periods that are good. 
	3. Do this for training and then test period.
5. **Residuals**
	1. Pick some time step to demonstrate what's going on in the strategy.  