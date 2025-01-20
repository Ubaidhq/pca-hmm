# Stock Forecasting using Hidden Markov Models

- Hidden Markov models the states are discrete but the observations can be either discrete or continuous.
- Not as much work has been done on hidden Markov models as opposed to other machine learning models.
- Hidden Markov Model has the Markov property in that the transition matrix from the current state is only dependent on the current state and nothing prior.  
- HMMs have been a powerful tool for analysing non-stationary systems.  Stock Markets are non-stationary systems and the observations are continuous in nature.
- In the example in this paper, we consider the state to be the STATE. 
- The observations are a vector of four quantities, specifically the daily close open, high and low.  
- We can assume the observations to be distributed as multivariate Gaussians.
- Each observation from one day to another is assumed to be independent but for each element within an observation there will obviously be a degree of correlation. 
- I think this assumption is rather far fetched as you would expect consecutive.

𝜆 = (𝐴, 𝜇, Σ, 𝑃 ) 
- The hidden Markov can be represented as above, with A the transition matrix with 𝐴 = [𝑎𝑖𝑗] where 𝑎𝑖𝑗 is the state transition probability from 𝑠_i to s_j.  
- What they are doing is quite simple.  They essentially have a lag of K and then go back K days from today and then compare the log likelihood of a sequence of length K which has previously occurred.  Whichever one is most similar is then used.  The price difference between the final day and the next day from this previous subsequence is then added to the current day to get a prediction of tomorrows price.  
- What we do after this is retune our model so that our sequence does not diverge.
- Would certainly be worth going over what AIC are and what BIC are.  
- They used the python library hmm learn, an open source python library to train the model and calculate the likelihood of the observations. 
- 

