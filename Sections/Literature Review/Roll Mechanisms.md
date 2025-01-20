A trader holding a futures contract can generally do one of three things: *offset* or close his position, *roll* the contract or take delivery of the underlying asset.  

 Rolling a contract is said to be when you sell (buy) a security to close a position and buy (sell) another (of the same type and with the same terms) with a longer period until expiration date (the roll forward decision) \cite{Roll strategy}.

 In commodities markets, the majority of investors do not want to take physical delivery.  Usually this would involve additional costs and work to store the product and ensure a certain degree of quality when proceeding to sell. 

 As a result, traders will often opt to roll their position into futures contracts with longer term maturity dates in order to maintain their exposure to the market without taking delivery.  

 When looking to roll contracts, traders are particularly concerned about *trade execution quality*, which is typically measured by the difference between the total revenue that would have been generated at the *arrival* price and the total revenue that is actually generated.  The difference is generally referred to as the *implementation shortfall* \cite{Perold 1988}.  With a basic understanding of market microstructure, it is easy to see why such an implementation shortfall should exist.  Ignoring any effects on how trades would affect the behaviour of other market participants, if an order size is greater than the volume at the best ask or best buy price, then once this is exhausted, the volume at the next best price will be consumed.  This will mean that a portion of the orders are not executed at the best ask or best buy prices.  
 
Furthermore, given the market's expectation of traders rolling contracts over, the expected execution quality leading up to a contract's maturity tends to be *periodic* in nature.  

 The roll mechanisms which look to exploit this can be broken down into two main categories: single-day and multiple-day.  Single-day mechanisms usually involve closing and then re-entering a futures position a fixed number of days before contract expiry.  Multiple-day mechanims instead are spread out over multiple days.  A common example is the *Goldman roll strategy*: the futures contract is *rolled* over between the fifth and ninth business days of the month proceeding the maturity month in a uniform fashion.  

 