## Framework for selecting number of hidden states

An important step in the process of building a HMM is to decide on the number of states.  Given that the *suitability* of a model is deemed entirely through the log-likelihood of observing the set of observations in the EM algorithm, it is trivial to imagine an extreme case where the number of states is equal to the number of observations.  

In essence, it is the aim of the modeller to build a model with an appropriate number of states to encapsulate the key themes within a given time series, but to not *overfit* and deny themselves of the opportunity to gather any meaningful results.  In other words, we seek the model that is most parsimonious.   

It is therefore a reasonable first step to atttempt to *penalise* an overly complex model whilst still crediting the log-likelihood.  The two most common *information criteria* which are used in practice are the *Akaike Information Criteria* (AIC) and *Bayesian Information Criteria* (BIC).

Despite these well established techniques, research as shown that they tend to favour models with a number of states that are undesirably large in situations where states shall be meaningful entities \cite{Pohle}.  

To combat this Pohle et al develop a pragmatic ordered selection process detailed below.  

### Step 1 
Decide *a priori* on models with a number of hidden states between a sensible range.  Without this restriction, the modeller increases the chance of an undesired selection biased.  

### Step 2
Apply an incremental approach to model selection, investigating the additional information that an additional state is able to uncover.  Examining state-dependent distributions and the Viterboi coded sequence at this stage is particularly useful. 

### Step 3
This is the model checking phase.  A model check based on psuedo-residuals \cite{Patterson et al, 2009} could for example reveal that a three-state model is preferred over a two-state model by AIC or BIC because the two-state model cannot capture the right tail of the empirical distribtuion. 

### Step 4
It is a sensible step to then examine model selection criteria such as AIC and BIC at this stage.  If there is a significant increase in an information criteria when increasing the number of state by one, this suggests that the model with extra states can better capture some underlying structure inherent within the model.  However, the structure itself may not always be of particular use in the context of the problem. 


### Step 5 
Combine the findings the previous 3 steps to pick one or two models. 

### Step 6
If the results obtained are not conclusive then it is wise to report results for each of the most plausible models.  

## Selecting number of hidden states
The range of hidden states that are considered in the first step are two to four states.  The motivation behind this is inspired by the fact that the individual observation series, log-returns and volatility, both often transition between two regimes themselves.  Whilst fairly noisy, the mean of log returns would be slightly higher in a bull market as opposed to in a bear market.  With regards to volatility, the majority of financial time series will have regimes of high and low volatility.  

In Step 2, we examine the empiricial distribution of the observations conditioned on their Viterbi decoded state, as well as the state-decoded observations themselves.  


------------------------------------------------------------------


## Information Criterion

### AIC


### BIC

The BIC is centered around comparing the posterior distribitions of candidate models.  The approach for model selecting is by comparing probabilities that each of the models under consideration is the true model that generated the obsereved data \cite{Kuha}.  

Suppose {$M_1, M_2, \cdots , M_k$} are a set of candidate models, and prior distributions $p(M_k)$ are assumed for each model.  We have $$p(D|M_k) = \int  p(D|\theta_k, M_k)p(\theta_k|M_k)d\theta_k$$
as the marginal likelihood of model $M_k$.  We also have $$ p(M_k|D) \propto p(D|M_k)p(M_k)$$
Now we can avoid having to calculate the proportionality constant in the above relation by looking at ratios as below: $$ \frac{p(M_2|D)}{p(M_1|D)} =\frac{p(D|M_2)}{p(D|M_1)} \cdot \frac{p(M_2)}{p(M_1)}$$
The middle term $$ \frac{p(D|M_2)}{p(D|M_1)}$$
is known as the *Bayes Factor*, and is considered to be a measure of evidence in facover of $M_2$ over $M_1$.  

### Similarities 

### Differences


Through various literature on financial time series and the state of the market, reoccuring regimes have been identified.  There is often a significant correlation between regimes and the state of the local or global economy.  For example, a *bull* market, which is commonly accepted to be when stock prices rise by 20% after two declines of 20% each, generally happens in line with a strengthening economy or one which is strong already.  Such a relationship makes logical sense.  A strenghthening economy fuels optimism in many market participants who then have a *bullish* view on the future.

Whilst using the *level* of the market to decipher hidden states is sensible, restricting the input to the model to this would mean we pay no direct attention to another key component of the market: volatility.


