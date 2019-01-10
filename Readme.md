# Risk Aware Top-k Recommendations
The algorithm to re-rank the recommendation list considering the risks associated with the expected payoff values
### Prerequisites
The following modules needs to be installed

```
    python2.7
    scikit-learn
    numpy
    scipy
    joblib

```

### Installing
```
pip2 install sklearn
pip install numpy
pip install scipy
pip install joblib

```
unzip the file and run the code

## Running the Code

To run the algorithm, execute the command
```

./main.py

```
The code expects the data directories to be stored under ../Datasets/ directory and the exact details of the ratings and meta-data files to be updated in data_utils.py file

```
##Results

The results are stored in the data directory.  Under the data directory, the 
``` result_%d.log ```
contains the values of all the performance metrics mentioned in the paper.
