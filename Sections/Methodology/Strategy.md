
## Training Phase
To test the strategy, we will use a benchmark PCA strategy without any regimes, or a single regime, and compare this to a PCA strategy with multiple regimes for both a sliding window of various length and an expanding window. 

In the sliding window case, we only let our model observe a snapshot of the time series at a fixed length, say a 2-year period.  Whereas in the expanding case, new data is continously added one by one as you move through time.  

An expanding window has advantage of allowing the model to pick up market dynamics that do not occur frequently enough to have a high chance of being seen in a given window.  For instance, a window beginning just after the 2008 financial crash would be capturing very different dynamics to one including the crash.  

However, on the flip-side, using data from the distant-past may not be reflective of recent-past market behaviour, which could potentially negatively impact the model's perfrmance.  

### Grid Search
We begin by deciding to split our data ino a training and test dataset.  A common ratio of 60:40 is chosen for this split.  

We then perform a grid search over 5 different holding periods as well as 3 different window lengths.  The holding periods are: 3, 5, 10 and 15 days respectively.  The holding periods are substanstially smaller than what has been used in other research such as \cite{Salomon}.  The reasoning for this is that we want allow was many trading days as possible without entering into a position whereby the contracts roll forward without the expiration of the holding period. 

Salomon also shows that using longer window lengths of 2 or more years produce more stable results and so window lengths of 2, 3 and 4 years are considered.  

The trading strategy is: we purchase all contracts with negative residuals and take a short position in all positive residuals.  The residuals are calculated by subtracting a reconstructed dataset produced by the first three principle components of the regime we are in from the actual value.  

We purchase a maximum of 100 contracts on any trading day, and weight the number of contracts by the ratio of the residuals for across all the contracts.  We then hold these two contracts for the number of holding days and sell.  If the regime that we are in changes before our holding period expires, we liquidate our position.  

The idea here is given that the residuals are *mean-reverting* we expect the prices of contract with positive residuals to drop and those with negative residuals to increase.

It is important to note that by purchasing contracts based on the residuals obtained from the first three princople components, our portfolio, when non-empty, should be neutral to movements in the futures curve.  

Given that the contracts roll forward on a single day, we want to avoid the scenario where we purchase a contracts which roll over before they are sold.  This could potentially distort the final results given that there would be a price change due to the roll.  For example, if futures contracts for Cocoa are marked for March, May, July, September and Decemnber, and if contracts roll over on the last day of the proceeding month, we would want to consider trading days at least as many days holding period before this.  

An important simplification over real-world trading here is that we assume there are no transaction or trading costs.  As a result of this, and for the fact that we are constructing a curve-neutral portfolio, one would theoretically expect such a strategy to be profitable.  In real-world conditions, it is often the case that transaction costs eat into the small profits that can be made from trading on the residuals.  Therefore, in order to measure performance, it is neccesary to set a bench mark to compare to.  The benchmark strategy will be a similar strategy in all aspects to the one described above except for the fact that there will be no regime-identifaction step.  At each time step, we will instead conduct PCA on all historic data before ge current date.

## Curve Neutral
As in \cite{Salomon}, the relative weighting of contracts in the butterfly position is important in maintaining neutrality to movements in the futures curve.  