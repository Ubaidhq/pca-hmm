-  Market conditions change over time leading to bullish or beaist market sentiments.
- Using the hidden markov models, we are better able to capture the stylized factors such as fat tails and volatility clustering compared with the standard Geometric Brownian motion. 
- Before financial market participants start to look for a trading opportunity, it is important they look at the current market situation and forecast subsequent market conditions so that they can decide on the appropriate trading strategies. 
- Since the mean and variance of the return in different market states is expected to be different, the forecast of return should be better in theory if the regime of the current economic situation can be figured out.
- The different number of market states should be considered and analysed by an information criterion like AIC, log likelihood and BIC.  
- Can we then have a hidden markov model for the differnet commodities and have some sort of aggregation? 
**- It is really important that we look into multiple observations because this is what we have in our case.** 
- These are the list of papers that I need to look at: 
	- *Rabiner 1989.*
	- *Samaria and Young (1994)*
	- *Aggoun and Moore (1995)*
	- *Elliott and Van der Hock (1997)*
	- Hardy (2001)
	- Roman et al (2010)

## Expectation Maximisation Algorithm
+ For parameter estimation, the standard approach is to find a maximum likelihood.  In Baum, Petrie and Weiss (1970), the Baum-Welch algorithm, also called EM algorithm is developed to estimate parameters in HMMs.  A gentle introduction of the EM algorithm adn ti's application to paramter estimation is provided by Bilmes (1998). 
+ Baggenstoss (2001) provides the modification of the Baum-Welch algorithm for hiden Markov models considering mutliple observation spaces.  

## Viterbi Algorithm 
+ Used for decoding.  
+ Viterbi algorithm is a dynamic programming algorithm which aims to find the most likely sqeuence of hidden states.  
+ Recursive optimal solution which aims to solve the problem of estimating the state sequence.  
+ Viberti (1967) the full description of the algorithm is provided.  
+ Fornet (1973) good introduction to the Viberti algorithm.

## Models and methods
+ The number of states is part of the model input/selection and aims to pick up the best modelt o fit the data and to analyse the market states. 
+ The hidden Markov model has fintite sets of state, which are associated with probablity distributions.  

## The Forward-Backward Algorithm
-  Need a pen and paper to work through and understand this.  The videos will be super helpful!  


# Plots
- 