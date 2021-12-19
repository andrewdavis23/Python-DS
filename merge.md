## merge_ordered()
Perform merge with optional filling/interpolation.
Designed for ordered data like time series data. Optionally perform group-wise merge (see examples).  

```python3
# Use merge_ordered() to merge gdp and sp500, interpolate missing value
gdp_sp500 = pd.merge_ordered(gdp, sp500, left_on='year', right_on='date', 
                             how='left',  fill_method='ffill')

# Subset the gdp and returns columns
gdp_returns = gdp_sp500[['gdp','returns']]

# Print gdp_returns correlation
print (gdp_returns.corr())
```
## merge_asof()
Sames as merge_ordered, but it was match on nearest, nearest below (behind), or nearest above (forward) values.
# Merge gdp and recession on date using merge_asof()
gdp_recession = pd.merge_asof(gdp,recession,on='date')

# Create a list based on the row value of gdp_recession['econ_status']
is_recession = ['r' if s=='recession' else 'g' for s in gdp_recession['econ_status']]

# Plot a bar chart of gdp_recession
gdp_recession.plot(kind='bar', y='gdp', x='date', color=is_recession, rot=90)
plt.show()

![image](https://user-images.githubusercontent.com/47924318/146685574-c7955503-bd08-4fe2-adf0-dd6b1d342e26.png)
