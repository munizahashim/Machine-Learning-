#importing pandas library
import pandas as pd
from pandas_profiling import ProfileReport
#Loading the dataset
mydata = pd.read_csv('dataset-1.csv')
print(mydata)
profile = ProfileReport(mydata)
profile.to_file(output_file="RepoetFile.html")