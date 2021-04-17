[DataCamp Course](https://campus.datacamp.com/courses/introduction-to-importing-data-in-python)

### Python

```python
# Read the directory file list
! ls
```
```python
# Open a file: file
file = open('moby_dick.txt', "r")

# Print it
print(file.read())

# Check whether file is closed
print(file.closed)

# Close file
file.close()

# Check whether file is closed
print(file.closed)
```
```python3
# Read & print the first 3 lines
with open('moby_dick.txt') as file:
    print(file.readline())
    print(file.readline())
    print(file.readline())
```
```python3
# Import pickle package
import pickle

# Open pickle file and load data: d
with open('data.pkl', 'rb') as file:
    d = pickle.load(file)

# Print d
print(d)

# Print datatype of d
print(type(d))
```

