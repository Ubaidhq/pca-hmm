# Abstract

*This paper proposes a framework to detect financial crises, pinpoint the end of a crisis in stock markets and support investment decision-making processes. This proposal is based on a hidden Markov model (HMM) and allows for a specific focus on conditional mean returns.*

- The aim is to focus on each market regime which is characterized by a probability distribution with a mean return and a volatility level that can help us to achieve relevant insights into the dyanmics and regime-switching of stock markets.  
- The two questions they want to answer:
	- Are we able to detect the end of a crisis and the switch to a stable period?
	- How can we support investment decision making proccesses during such periods? 
- Quite suprisingly. we find evidence that it performs better than a main reference in stochaswtic volatility modelling such as the threshold GARCH model with student-t innovations? 
- **What is a GARCH model?**
	- generalised autoregressive conditional heteroskedasticity process.
	- There are several forms of GARCH modeling. Financial professionals often prefer the GARCH process because it provides a more real-world context than other models when trying to predict the prices and rates of financial instruments.
	- The model allows for better capturing of volatility clustering which is a very commonly observed phenomenon in markets. 
	- $\sigma_t^2=w+\alpha x_{t-1}^2 + \beta \sigma_{t-1}^2$ 
	- alpha is the volatility's spikiness.  How quick are it's reactions.
	- beta is volatilitys persistency.  How long to change. 
- Stable periods, crisis, and financial bubbles should be characterised by significantly different mean returns, and our analysis reveals that in the US stock market, the conditonal means differ statistically significantly across time periods. 
- We, therefore, undertake an endogenous detection of different market phases, which contributes to extant research involving Markov-switching models that usually identify the number of market regimes a priori, often predetermining the number of latent states that characterize the unob- servable Markov chain. For example, a study might select two predetermined latent states that represent a low-volatility regime (i.e. the bull market phase) and a high-volatility regime (i.e. the bear market phase) [22,23,28].
- [[Identify Bull and Bear Markets in Stock Returns]]