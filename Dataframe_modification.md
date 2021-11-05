#### Change index and column names 

```python3

# Assign the string 'MONTHS' to sales.index.name
sales.index.name = 'MONTHS'

# Print the sales DataFrame
print(sales)

# Assign the string 'PRODUCTS' to sales.columns.name 
sales.columns.name = 'PRODUCTS'

# Print the sales dataframe again
print(sales)

##RESULT:

#PRODUCTS  eggs  salt  spam
# MONTHS                    
# JAN         47  12.0    17
# FEB        110  50.0    31
# MAR        221  89.0    72
# APR         77  87.0    20
# MAY        132   NaN    52
# JUN        205  60.0    55
```

#### Slice by index

```python3

# Use Boolean conditions to subset temperatures for rows in 2010 and 2011
temperatures_bool = temperatures[(temperatures['date'] >= '2010-01-01') & (temperatures['date'] <= '2011-12-31')]
print(temperatures_bool)

# Set date as the index and sort the index
temperatures_ind = temperatures.set_index('date').sort_index()

# Use .loc[] to subset temperatures_ind for rows in 2010 and 2011
print(temperatures_ind.loc['2010':'2011'])

# Use .loc[] to subset temperatures_ind for rows from Aug 2010 to Feb 2011
print(temperatures_ind.loc['2010-08':'2011-02'])

```

#### Indexing multiple levels of a MultiIndex

```python3

# Look up data for NY in month 1 in sales: NY_month1
NY_month1 = sales.loc['NY',1]

# Look up data for CA and TX in month 2: CA_TX_month2
CA_TX_month2 = sales.loc[(['CA', 'TX'], 2),:]

# Look up data for all states in month 2: all_month2
all_month2 = sales.loc[(['CA','NY','TX'], 2),:]

#              eggs  salt  spam
# state month                  
# CA    1        47  12.0    17
#       2       110  50.0    31
# NY    1       221  89.0    72
#       2        77  87.0    20
# TX    1       132   NaN    52
#       2       205  60.0    55

print(CA_TX_month2,all_month2)

#              eggs  salt  spam
# state month                  
# CA    2       110  50.0    31
# TX    2       205  60.0    55              
#              eggs  salt  spam
# state month                  
# CA    2       110  50.0    31
# NY    2        77  87.0    20
# TX    2       205  60.0    55

```

# Modify columns

```python3

# Create indiv_per_10k col as homeless individuals per 10k state pop
homelessness["indiv_per_10k"] = 10000 * homelessness['individuals'] / homelessness['state_pop'] 

# Subset rows for indiv_per_10k greater than 20
high_homelessness = homelessness[homelessness['indiv_per_10k']>20]

# Sort high_homelessness by descending indiv_per_10k
high_homelessness_srt = high_homelessness.sort_values('indiv_per_10k',ascending=False)

# From high_homelessness_srt, select the state and indiv_per_10k cols
result = high_homelessness_srt[['state','indiv_per_10k']]

```

# Aggregate Funcs

```python3

# Import NumPy and create custom IQR function
import numpy as np
def iqr(column):
    return column.quantile(0.75) - column.quantile(0.25)

# Update to print IQR and median of temperature_c, fuel_price_usd_per_l, & unemployment
print(sales[["temperature_c", "fuel_price_usd_per_l", "unemployment"]].agg([iqr,np.median]))

temperature_c  fuel_price_usd_per_l  unemployment
iqr            16.583                 0.073         0.565
median         16.967                 0.743         8.099

```

#### Duplicates, Counting

```python3

# Drop duplicate store/type combinations
store_types = sales.drop_duplicates(subset=["store", "type"])

# Drop duplicate store/department combinations
store_depts = sales.drop_duplicates(subset=["store", "department"])

# Count the number of stores of each type
store_counts = store_types['type'].value_counts()
print(store_counts)

# Get the proportion of stores of each type
store_props = store_types['type'].value_counts(normalize=True)
print(store_props)

# Count the number of each department number and sort
dept_counts_sorted = store_depts['department'].value_counts(sort=True)
print(dept_counts_sorted)

# Get the proportion of departments of each number and sort
dept_props_sorted = store_depts.department.value_counts(sort=True, normalize=True)
print(dept_props_sorted)

```

