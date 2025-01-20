- We are going to use the hmmlearn package to build the hidden markov model which serves as the method for identifiying the regimes and implementing the regime switching model.
- There are three primary parameters that we need to define:
	- The number of hidden states
	- The covariance type
	- The threshold for the maximum number of iteratiosn to perform for the expectation maximization algorithm.


- We have our three derived variables.  
- We want to input these into the hmm. 


- Are we going to create a python package for the use of the multiple observation case that we have described? 
- Can we deal with the multiple observations without making any assumptions and just accepting that there is going to be a higher dimensionality than we probably would have wanted?  I think this is the way we are going to have to go.
- It would be sick to implement a python package for multiple observations yes, but we don't even know if there is merit in what we are trying to do.  
- Imagine coming up with this python package to help identify regimes better, which won't be crazy because the final PCA trading strategy really doesn't care how the regimes have been indentified. 


## Coding from scratch

T - length of observation sequence
N - number of hidden states
M- number of observables
Q - hidden states
V - set of possible observations/alphabet
A - state transition matrix
B- emission probability matrix
pi - initial state probability distribution
O - observation sequence
X - hidden state sequence

### Forward Algorithm
- In the single observation case, we define the function $\alpha$ to be
		$$\alpha_t(i) =P(O_1, O_2, ... O_t, q_t = S_i|\lambda)$$
- The steps for the algorithm are as follows: 
$$
	\alpha_1(i) = \pi_i b_i(O_1) $$

$$	\alpha_{t+1}(j) = ( \sum^N \alpha_{t}(i) a_{ij} ) b_j(O_{t+1}) $$

	$$ P(O|\lambda) = \sum^N \alpha_T(i) $$
![[Screenshot 2022-03-15 at 09.19.34.png]]

- The algorithm is like a chain of probabilities that you can compute at each step.
- It's easy to see why it can be viewed as a DP problem because at each step you need the results of the previous steps and caching these would make calculations a lot faster. 

### How does this then compare to the multiple observation case?

Consider a set of observations, where $$O = \{ O^{(1)},O^                                                      {(2)}, \cdots, O^{(K)}\} $$
and each $O^k$  $$O^{(k)} = \{o^{(k)}_1, o^{(k)}_2, \cdots, o^{(k)}_{T_k} \}$$
Recall there are three problems: 
1. Finding the likelihood of a set of observations given the model - here we use the forward-backward algorithm
2. Finding the *best* sequence of states given the parameters of a model and the sequence of observations
3. Given ***only*** the sequence of observations, estimate the parameters of the model.  

- We have been given the observations, which are our derived variables.  Therefore we can implement problem 3 fine, but then how do we deal with problems 1 and 2?
	- We still only have the single observation sequence, so we can use this to tranverse through the lattice.  
	- At each time step, instead of just having the one emission probability, we now have 3, which are not independent.  
	- The symbol emisssion probability $$  b_n(m) = \frac{\sum_{k=1}^K \omega_k P(O^{(k)}|\lambda) \sum^{T_k}_{t=1, o^{(k)}_t = v_m} \gamma^{(k)}_t(n)}{\sum^K \omega_k P(O^{(k)}|\lambda) \sum^{T_k} \gamma^{(k)}_t(n)}$$
	- We are summing over all observation sequences, since we have $$O = \{ O^{(1)},O^                                                      {(3)}, O^{(3)}\} $$
	- What is the gamma function? $$\gamma_t(i) = P(q_t = S_i | O, \lambda)$$
	- which is the probability of being in state $S_i$ at time $t$ given the set of observations and the model.
	- N is the number of observations. 
	- I don't think the approach above is as intuitive? 
		- $o_t^{k} = v_m$, so we are summing over all observations from time t=1 to time $t=T_k$ - 
		- $b_n(m)$  - this is the emission probability of observation at any time being $v_m$ from the alphabet of possible observations given that we are in state n.
			- So in our case, we have 3 variables, so 3 observation sequences.
			- Then for each observation sequence, in the numerator,  we have: $$ \omega_{k^*} \cdot P(O^{(k^*)}|\lambda) \sum^{T_{k^*}}_{t=1, o^{(k^*)}_t = v_m} \gamma^{(k^*)}_t(n)$$
			- Are we updating at each time step? Or after all the observations have been seen? 
				- After all the observations have been seen. 
			- What is the standard update to the emission probability? ![[Screenshot 2022-03-20 at 19.13.51.png]]
- This is really going to be an issue since we are not dealing with discrete quantities.  But let's try anyways?  We could categorise our variables? If we categorise slope and volatility and neglect level then we have a much easier way, but then we essentially only have 9 combianations right?
	- Ah let's not over complicate it here.
	- This may actually make much more sense because we are going to get a zero emission probability or a very low emission probabillity many times.  

### Categorising numerical derived variables
````python
import pandas as pd

df = pd.DataFrame({'urbanrate':[10,20,25,30,40,75,80,100]})
print (df)
   urbanrate
0         10
1         20
2         25
3         30
4         40
5         75
6         80
7        100

bins = [0, 24.999, 74.999, 1000]
group_names = [1,2,3]
df['urban'] = pd.cut(df['urbanrate'], bins, labels=group_names)
print (df)
   urbanrate urban
0         10     1
1         20     1
2         25     2
3         30     2
4         40     2
5         75     3
6         80     3
7        100     3
````