# Hmmlearn python package.  
π - initial probabilities? 
A - our transition matrix
θ - parameters of distribution(s) for observable.  

There are three fundamental problems for HMMs: 

* Given the model parameters and observed data, estimate the optimal sequence of hidden states.  
* Given the model parameters and observed data, calculate the model likelihood. 
* Given just the observed data, estimate the model parameters.  

The first two can be solved by the dynamic programming algorithms known as the Viterbi algorithm and the forward-backward algorithm, respectively.  The last one can be solved by an interactive Expectation-Maximization (EM) algorithm, known as Baum-Welch algorithm.

