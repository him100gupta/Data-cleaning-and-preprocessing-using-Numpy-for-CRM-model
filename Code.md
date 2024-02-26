**Importing the library**
``` Python
import numpy as np
np.set_printoptions(suppress = True, linewidth = 100, precision =2)
```
**Importing the data**
``` Python
dataset = np.genfromtxt("loan-data.csv", delimiter = ';', skip_header = 1, autostrip = True)
dataset
```
**Checking for the number of missing values**
``` Python
np.sum(np.isnan(dataset))
```
### Creating temporary arrays for later
``` Python
temp_fill = np.nanmax(dataset) + 1
temp_mean = np.nanmean(dataset, axis = 0)
temp_stats = np.array([np.nanmin(dataset, axis = 0),
                      temp_mean,
                      np.nanmax(dataset, axis = 0)])
```
### Creating arrays to get the indices of the numeric columns and text columns. 
**Array containing the indices of text columns**
``` Python
str_column = np.argwhere(np.isnan(temp_mean)).squeeze()
str_column
```
**Array containing the indices of the numeric columns**
``` Python
num_column = np.argwhere(np.isnan(temp_mean) == False).squeeze()
num_column
```
### Reimport the data into 2 dataset using the "str_column" & "num_column".
**This code will import the text columns only and store them in dataset_str**
``` Python
dataset_str = np.genfromtxt("loan-data.csv", delimiter = ';', skip_header = 1, 
                            autostrip = True, usecols = str_column, dtype = str)
dataset_str
```
**This code will import the numeric columns only and store them in dataset_num**
``` Python
dataset_num = np.genfromtxt("loan-data.csv", delimiter = ';', skip_header = 1, 
                            autostrip = True, usecols = num_column, filling_values = temp_fill)
dataset_num
```
### Import the header and process it.
**This code will import the header**
``` Python
Header_full = np.genfromtxt("loan-data.csv", delimiter = ';',autostrip = True, skip_footer = dataset.shape[0],dtype = str)
Header_full
```
**This code will separate the numeric column header and text column headers into two different variables **
``` Python
header_strings, header_numeric = Header_full[str_column], Header_full[num_column]
```
## Now we will clean and preprocess the text columns individually
**Column "issue_date"**
``` Python
header_strings[0] = "issue_date"
np.unique(dataset_str[:,0])
dataset_str[:,0] = np.chararray.strip(dataset_str[:,0], "-15")
np.unique(dataset_str[:,0])
```
``` Python
months = np.array(['','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
```
This code will replace the month's name with the number as numbers take less memory compared to stings.

It iterates over each index (from 0 to 12) in the range of 13.
For each index 'i', it checks if the element in the first column of 'dataset_str' matches the abbreviated month name at index 'i' in the 'months' array.
If there is a match, it replaces that element with 'i', which is the numerical representation of the month.
If there is no match, it leaves the element unchanged.
``` Python
for i in range(13):
    dataset_str[:,0] = np.where(dataset_str[:,0] == months[i],
                               i,
                               dataset_str[:,0])
dataset_str[:,0]
```
**Column Loan status**
``` Python
dataset_str[:,1]
np.unique(dataset_str[:,1])
```
The column loan status have 8 unique values and blank values.

We can put this status into two categories namely Good and Bad.
The good category includes, 'current', 'Fully Paid', 'In Grace Period', 'Issued' and 'Late (16-30 days)'. While the bad constists of the leftovers.
"1" for good and "0" bad status
``` Python
status_bad = np.array(['', 'Charged Off', 'Default', 'Late (31-120 days)'])
dataset_str[:,1] = np.where(np.isin(dataset_str[:,1], status_bad),0,1)
np.unique(dataset_str[:,1])
```
**Column Term**
``` Python
np.unique (dataset_str[:,2])
dataset_str[:,2] = np.chararray.strip(dataset_str[:,2], " months")
header_strings[2] = "Term months"
np.unique(dataset_str[:,2])
```
Have missing data. Fill this with 60(considering the worst scenario) as this dataset is for Credit Risk modeling.
``` Python
dataset_str[:,2] = np.where(dataset_str[:,2] == '',
                            '60',
                            dataset_str[:,2])
np.unique(dataset_str[:,2])
```
**Column grade and subgrade**
``` Python
np.unique(dataset_str[:,3])
```
Have 7 unique categories in the grade column
``` Python
np.unique(dataset_str[:,4])
```
Have 36 unique values in the subgrade column including blanks.

The subgrade column is filled using the grade column and the lowest value is assigned to that garde.
``` Python
for i in np.unique(dataset_str[:,3])[1:]:
    dataset_str[:,4] = np.where((dataset_str[:,4] == '') & (dataset_str[:,3] == i), i + '5',
                                          dataset_str[:,4])
np.unique(dataset_str[:,4], return_counts = True)
```
Still, there are 9 missing values in the subgrade column. We can delete these rows but instead of deleting them, we can assign them the lowest grade which is "G5" as these data are for CRM
``` Python
dataset_str[:,4] = np.where(dataset_str[:,4] == '',
                             'G5',
                             dataset_str[:,4])
np.unique(dataset_str[:,4], return_counts = True)
```
The grade can be known from subgrade so, we can delete the column "grade" and it's header
``` Python
dataset_str = np.delete(dataset_str, 3, axis = 1)
header_strings = np.delete(header_strings, 3)
```
**Column verification status**
``` Python
header_strings
np.unique(dataset_str[:,4])
```
The 'blank' and 'Not verified' statuses are replaced with 0 while the 'Verified' and 'Source Verified' are replaced with 1
``` Python
dataset_str[:,4] = np.where((dataset_str[:,4] == '') | (dataset_str[:,4] == 'Not Verified'),0,1)
np.unique(dataset_str[:,4])
```
**Column URL**
``` Python
dataset_str[:,5]
dataset_str[:,5] = np.chararray.strip(dataset_str[:,5], "https://www.lendingclub.com/browse/loanDetail.action?loan_id=")
dataset_num[:,0]
```
It looks like the loan id we got from url is the same as the id number of the customer.
``` Python
np.array_equal(dataset_num[:,0].astype(dtype = np.int32), dataset_str[:,5].astype(dtype = np.int32))
```
They are same, so, delete this column as it's of no use in analysis
``` Python
dataset_str = np.delete(dataset_str,5, axis = 1)
header_strings = np.delete(header_strings,5)
```
**Column addr_state**
``` Python
header_strings[5] = "State Address"
np.unique(dataset_str[:,5], return_counts = True)
```
we have 49 unique states and 500 missing values.
THE US has 50 states out of which Iowa is missing. One possibility could be it was picked as a benchmark.
We can fill in the missing value with Iowa (IA).
``` Python
dataset_str[:,5] = np.where(dataset_str[:,5] == '',
                             'IA',
                             dataset_str[:,5])
np.unique(dataset_str[:,5], return_counts = True)
```
**Now we have done cleaning and processing the text data we can create a checkpoint**
``` Python
def checkpoint(file_name, checkpoint_header, checkpoint_data):
    np.savez(file_name, header = checkpoint_header, data = checkpoint_data)
    checkpoint_variable = np.load(file_name + ".npz")
    return(checkpoint_variable)
```
``` Python
checkpoint_strings = checkpoint("checkpoint-strings", header_strings, dataset_str)
```
Just to ensure the data have been same
``` Python
np.array_equal(checkpoint_strings['data'], dataset_str)
np.array_equal(checkpoint_strings['header'], header_strings)
```
## Now we will clean and preprocess the numeric columns
**Check missing values**
``` Python
np.sum(np.isnan(dataset_num))
```
There isn't any missing values as  we temporarily filled the numeric data with max+1 and imported the dataset, now we will replace all those temporarily fills with the actual values.

**Column id**
``` Python
np.isin(dataset_num[:,0],temp_fill).sum()
```
In the 1st column, we don't have any temporary filled values. So, no changes are necessary.

**Column funded amount**

For this column we will replace the temp fill with the minimum value of this column
``` Python
dataset_num[:,2]
dataset_num[:,2] = np.where(dataset_num[:,2] == temp_fill,
                           temp_stats[0, num_column[2]],
                           dataset_num[:,2])
dataset_num[:,2]
```
**Column loaned Amount, Intrest rate, Total payment, Installment**

Will replace the temp values with the maximum value of each column respectively.
``` Python
for i in [1,3,4,5]:
    dataset_num[:,i] = np.where(dataset_num[:,i] == temp_fill,
                           temp_stats[2, num_column[i]],
                           dataset_num[:,i])
```
## Combining the datasets
**Combining the numeric and text data**
``` Python
dataset_str.shape
dataset_num.shape
np.hstack((dataset_num, dataset_str)).shape
final_data = np.hstack((dataset_num, dataset_str))
```
**Combining the numeric header and the text header**
``` Python
final_header = np.concatenate((header_numeric, header_strings))
```
## Combining the dataset and the header to get the final datasheet
``` Python
final_loan_datasheet = np.vstack((final_header, final_data))
```
## Exporting the final datasheet as a ".csv"
``` Python
np.savetxt("final_loan_data.csv",
           final_loan_datasheet,
          fmt = "%s",
          delimiter = ',')
```
