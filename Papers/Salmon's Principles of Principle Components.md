+ After extensive historical simulations, Butterfly Model using PCA on yield levels as the weighting method is preferable to PCA on yield changes.  
+ Investors undertake butterfly trades for two main reasons (1) to take on market direction or slope eposure within a cash and/or duration constraint, or (2) to undertake a curvature or relative-value trade. 
	+ In the second scenario, the relative-value investor desires to take advantage of the relative cheapness or richness of one security (center) compared to two securities (wings) around it, without taking a market directional view.  The expecation is that the middle leg will come back in line with the wings.
+ Two most common butteryfly weightings are: (1) a 50-50 weighting (putting 50% of the duration-dollars in teh short wings and 50% in the long win).  (2) a cash-neutral and duration-neutral weighting. 
+ In order to structure market-neutral butterflies for pure relative-value plays, investors look toward statistical methods, such as regression analysis or volatility weighting.
+ To construct a butterfly-weighting scheme that produces a pure market-neutral curvature play, we use PCA on the yields of the 3 legs. 
+ PCA will identify the weightings to best immunize a butterfly trade to both market-level and slope factors - the first-two principal components - while leaving the exposure to curvature(or relative value), which is the residual variation among th threeyields.  Furthermore, because of a fundamental property of PCA, the PCA weighted butterfly spraed is not correlated with market level and slope.  This characteristic is the key to constructing curve-neutral butterfly trades.

## Levels or Changes? 
+ Better to conduct PCA on levels for butterfly trades rather than on changes to generate trading signals and to compute the trade weights. 
+ When going into a butterfly trade, we are primarily interested in taking advatange of anomalies in the yield levels fo the three securities.  That suggests using levels as the starting point.  But here is the puzzle: once we decide that the yield levels present a trading opportunity, you want to set up a curve neutral trade and hedge against yield level shifts (CHANGES) and slop changes.  This would suggest instead using the statistics of the yield changes to weight the trade rather than the statistics of the level. 
+ Unforunately trade weights obtained from the levels compared to the changes do not match up. 
+ The level weights may not hedge against incremental changes in level and slope, but can hedge against the aggreaget changes in these quantities.  By construction, the PC-weighted butterfly spread structured from levels will not be correlated with yield level and slope.  
+ This means that as the butterfly spread sweeps from one extreme to another, the short term changes in spread, slope and level will be wiped out.  
+ As a result, using levels for the PCs is the best way to come up with a curve neutral butterfly strategy.  

### How much historical data should we use? 
+ Using a very long time period appears to be the best.  However, there could be some legitimate concerns with this approach.  
+ Are the old data still relevant to the market today?  Should we not put more weight on recent data, as they probabyl better capture any effects of the current monetary policy?
+ However, by using data from a short period that reflects a specific monetary policy regime and economic conditions, one implicity assumes the continuation of the same environment going forward.  The yield statistics obtained that way may be completekly off in teh case of a policy change.  
+ In contrast, a longer time period will likeyl include cycles of tightening as well as easing, strong econimic growth as well as slowdown.  