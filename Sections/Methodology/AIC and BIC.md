
## Nguyen's work
- Uses AIC, BIC, HQ and BCAIC to determine an optimal nubmer of states for the HMM.
- Choosing a number of hidden states for the HMM is a critical task.
- These criteria are suitable for the HMM because, in the model training algorithm, the Baum-Welch algo is used to maximise the log-likelihood of the model.  
- $$AIC = -2 \ln(L) + 2k $$
$$ BIC = -2 \;ln(L) + k\ln(M)$$ $$HQC = -2 \ln(L) + k\ln(\ln(M))$$
$$ CAIC = -2 \;ln(L) + k(\ln(M) + 1)$$
L is the likelihood function for the model, M is the number of observation points and K is the number of estimated parameters within the model.

## Auto - HMM
- Train ratio - fraction of the data that is used for training.
- Cov_type - Can be full, diag etc
- Max_state - You want to cap the number of hidden states. 
- Iter - how many times the markov model will run.
- N - number of observations
- T - number of columns
- Flag - best observation corresponds to higher estate?
- We can just fork what he's done and build upon it using the following guidance from a paper.  

## AIC/BIC and HMM
- In general, when building models in which we are looking to optimise a likelihood function, the Akaike Information Criterion (AIC) and Bayesian Infomation Criterion (BIC) are used. 
- However, especialy in the ecological literature it has been demonstrated that using methods such as AIC and BIC can lead to a selection of many more states than expected apriori \cite{POHLE}.
- In addition to the features that actually motivate the use of state-switching models, such as multimodality and autocorrelation, real data often exhibit further structure, such as outliers and seasonality.  Therefore, by incorporating more states, there is the possbility that these structures are picked up by the model.  A high number of states can often be deemed unreasonable, particularly when the hidden state sequence and it's general dynamics are the focus of an analysis.
- Pohle et al in \cite{Pohle} propose a comprehensive methodology for evaluating the use of information criteria such as AIC and BIC to choose an appropriate number of hidden states.  

## Guidance
### What assumptions are needed to conduct AIC and BIC? 
1. Decide a priori on the candidate models, in particular the minimum and maximum number of states that seem plausible, and fit the corresponding range of models.
	1. This is fairly straight forward in our case. 
2. Closely inspect each of the fitted models, in particular by plotting their estimated state-dependent ditributions and by considering their Viterbi-decoded state sequences. 
	1. So we want the mixture/gaussian distributions of each state.
		1. For example, we could have 2 states where one has a mulinomial distribution with two peaks, but then when we add another state, what happens is we now get that multinomial distribution split into it's two constituent Gaussian components for each corresponding state. 
3. We can use model checking methods, in particular residual analysis, to obtain a more detailed picture of the fitted models, and to validate or invalidate any given candidate model. 
	1. What would this tell us exactly? 
		1. Validation of HMMs via model checking is covered in detail in chapter 6 in Zucchini et al. 
		2. In order to make an informed choice on the number of states, it is important to understand what causes the potential preference for models with many states. 


### Mixture Emissions
Remember that the Gaussian Mixtures concern the makeup of the probability distributions of each state and not the model as a whole. 

*Assuming that a single Gaussion emission distribution is accurate and representative enough to model the porbability of observation vectors of any state of a HMM is often a very strong and naive one.  Instead, a more powerful approach is to represent the emission distribution as a mixture of multiple multivariate Gaussian densities.  *

![[Screenshot 2022-03-27 at 20.14.05.png]]