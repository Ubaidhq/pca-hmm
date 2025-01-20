# Hamilton 1990/1994 original work on HMM for Regime Switching.  

* What even is an autoregression?  
	* Using past values to predict future ones. 

## Statistical analysis of mixture distributions

* Smoothing - smoothing is the joint distribution of all the thetas given all the observations up until now.  
* Filtering - is the distribution of theta_n given all the observations up until now. 
* Essentially you have different distributions and then can use maximum likelihood estimation to get a new estimation of what the parameters are.  

## Time series models of changes in Regime

* We now want to build a model that is flexible enough to follow more than one time series (our regimes) over different subsamples.  
* The proposal is to model the regime as the outcome of a Hidden Markov Model.
* Why might the Hidden Markov Model be suitable for the regime switching model?
	* We can encapsulate the one state or deterministic scenario by having a ‘final’ state.  
* Flexible enough to allow us to forecast with the regime change included. 
* The algorithm then essentially involves sampling some observations, then updating the parameters, then sampling some more observations using this updated model etc etc.  (I’m not really clear on this, might need to go through it again) 

