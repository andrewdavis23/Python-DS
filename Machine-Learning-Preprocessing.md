```python
## Sometimes the necessary modules are not shown as being imported.

## Stratifying #############################################################################################################
# Class: the output category of data
# Stratify:  Make the original dataset class distribution match the class distribution in both the training and test class

# Create a data with all columns except category_desc
volunteer_X = volunteer.drop("category_desc", axis=1)

# Create a category_desc labels dataset
volunteer_y = volunteer[["category_desc"]]

# Use stratified sampling to split up the dataset according to the volunteer_y dataset
X_train, X_test, y_train, y_test = train_test_split(volunteer_X, volunteer_y, stratify=volunteer_y)

# Print out the category_desc counts on the training y labels
print(y_train["category_desc"].value_counts())


## KNN & Normalization #####################################################################################################

#### X (Proline is highly variable):
#   Proline  Total phenols   Hue  Nonflavanoid phenols
#0     1065           2.80  1.04                  0.28
#1     1050           2.65  1.05                  0.26
#2     1185           2.80  1.03                  0.30
#3     1480           3.85  0.86                  0.24
#4      735           2.80  1.04                  0.39 

# Split the dataset and labels into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y)

# Fit the k-nearest neighbors model to the training data
knn.fit(X_train,y_train)

# Score the model on the test data
print(knn.score(X_test,y_test))

# output is between 0.6 and 0.8

#### LOG NORMALIZATION

# Print out the variance of the Proline column
print(wine['Proline'].var())

## 99166.71735542428

# Apply the log normalization function to the Proline column
wine['Proline'] = np.log(wine['Proline'])

# Check the variance of the normalized Proline column
print(wine['Proline'].var())

## 0.17231366191842018

## Standardization ##########################################################################################################

# GENERAL FORM
# for sklearn.preprocessing = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)

# EXAMPLE

# Import StandardScaler from scikit-learn
from sklearn.preprocessing import StandardScaler

# Create the scaler
ss = StandardScaler()

# Take a subset of the DataFrame you want to scale 
wine_subset = wine[['Ash', 'Alcalinity of ash', 'Magnesium']]

# Apply the scaler to the DataFrame subset (this is a numpy array)
wine_subset_scaled = ss.fit_transform(wine_subset)

## Fix accuarcy of model with SCALING (see KNN example above where accuracy = (0.6,0.8))

# Create the scaling method.
ss = StandardScaler()

# Apply the scaling method to the dataset used for modeling.
X_scaled = ss.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)

# Fit the k-nearest neighbors model to the training data.
knn.fit(X_train,y_train)

# Score the model on the test data.
print(knn.score(X_test,y_test))

# New accuracy is within (0.88,1.0)

## Feature Engineering (Column Editing) ##################################################################################################

##### TRANSFORM - uses mean and variance, in this case it's transforming a BOOL column

# Set up the LabelEncoder object
enc = LabelEncoder()

# Apply the encoding to the "Accessible" column
hiking["Accessible_enc"] = enc.fit_transform(hiking["Accessible"])

# Compare the two columns
print(hiking[["Accessible", "Accessible_enc"]].head())

    # Accessible  Accessible_enc
    # 0          Y               1
    # 1          N               0
    # 2          N               0
    # 3          N               0
    # 4          N               0
    
##### Dummie - returns DF where categories are column and index matches original index. Values are 0 and 1. Basically a truth table for categories.

# Transform the category_desc column
category_enc = pd.get_dummies(volunteer["category_desc"])

# Take a look at the encoded columns
print(category_enc.head())

###### Modify Column with REGEX

# Write a pattern to extract numbers and decimals
def return_mileage(length):
    pattern = re.compile(r"\d+\.\d+")
    
    # Search the text for matches
    mile = re.match(pattern, length)
    
    # If a value is returned, use group(0) to return the found value
    if mile is not None:
        return float(mile.group(0))
        
# Apply the function to the Length column and take a look at both columns
hiking["Length_num"] = hiking["Length"].apply(lambda row: return_mileage(row))
print(hiking[["Length", "Length_num"]].head())

```
