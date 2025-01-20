
# The Forward Backward algorithm
Esentially the probability of interest here is $P(O| \lambda)$, where $O$ is the sequence of observations.  Namely, what is the probability of such an observation sequence $O$ given the model $\lambda$. A common method to figure out what this probability is to first look at $P(O, Q| \lambda)$, and then marginalise over all possible sequences, $Q$.  Therefore we want to find 

							$\sum_{all\ Q}P(O, Q|\lambda)$

Note that $P(O, Q|\lambda) =P(O|Q, \lambda)P(Q|\lambda)$.  Therefore, to find the joint probability of the observations and sequences given the model, the probablity of the observations given the sequences and the model and the probability of a given set of state sequences given the model is sufficient.

By assuming the statistical independence of observations (a natural step once the Markov property is assumed). the probability of a sequence of observations given the sequence of hidden states and the model can be calculated in the following way: 

$P(O|Q.\lambda) = \prod P(O_t|q_t, \lambda)$,

where $q_t$ is the state at time $t$ of a fixed state sequence $Q = q_1, q_2, \cdots, q_T$.

What we find with this marginalisation process for $P(O,Q| \lambda)$, is that calculation can very quickly become extremely computationally expensive and infeasible \cite{rabiner}.  A more efficient algorithm, the *forward backward algorithm* is first devised in \cite{Baum1967|}.  The algorithm is a dynamic programming algorithm and uses an inductive process to tackle the problem.  Note that only the forward part of the algorithm is neccessary to efficiently compute and solve problem 1.  The backward problem is introduced here, but is used in the solving of problem 3. 

To begin, first define $\alpha_t(i) =P(O_1, O_2, ... O_t, q_t = S_i|\lambda)$ to be the probability of the observations up until time t whilst also being in state $S_i$ at time 

## The Viterbi Algorithm

The second problem of finding the *best* sequence of states given the observations and the model is slightly more nuanced and highly depdent on how we define the *best* or most optimal sequence of states.  




