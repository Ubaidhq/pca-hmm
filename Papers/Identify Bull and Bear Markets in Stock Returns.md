# Abstract
Article uses a Markov-switching model that incorporates duration dependence to capture non linear structure in btoh the conditonal mean and conditional variance of stock returns.

- The model sorts returns into a high-return stable state and a low-return volatile state.  We label these as bull and bear markets respectivelty.  The filter identifies all maor stock-market downturns in over 160 years of monthly data.  
- The HMM model sorts data endogenously into regimes. 
- The Markov-switching model was shown by [[Ryden, Terasvirta, and Asbrink (1998)]] (this paper is a little bit too technical for use in this report) to be well suited to explaining the temporal and distributional properties of stock returns.  
- The Markov-switching model has been used extensively in modelling nonlinear structure in time series data.  For example: 
	- [[Turner, Startz and Nelson (1989)]] use the model to account for a time-varying risk premium in stock returns.
	- [[Schaller and van Norden (1997)]] used the aproach to distringuish between fads and bubbles in stock returns.  
	- [[Hamilton and Lin (1996)]] used the model to capture the nonlinear dynamics in the stock market and business cycle.  
	- [[Hamilton's (1989) ]]first-order Markov model will not capture duration dependence ins tates.  The latter could be particularly important in explaining volatility clustering, mean reversion, and nonlinear cyclical features in returns. 
	- Ignoring duration depedence could result in a failure to capture important properties of stock returns.
	- [[Durland and Mc-Curdy (1994)]] developed a parsiminious implementation of a a higher-order Marov chain that allowed state transition probabilities to be duration dependent.  