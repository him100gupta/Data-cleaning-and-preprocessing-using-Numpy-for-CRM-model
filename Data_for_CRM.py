#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
np.set_printoptions(suppress = True, linewidth = 100, precision =2)


# # Importing Dataset

# In[2]:


dataset = np.genfromtxt("loan-data.csv", delimiter = ';', skip_header = 1, autostrip = True)
dataset


# # Checking for missing data

# In[3]:


np.sum(np.isnan(dataset))


# In[4]:


temp_fill = np.nanmax(dataset) + 1
temp_mean = np.nanmean(dataset, axis = 0)


# In[5]:


temp_mean


# In[7]:


temp_stats = np.array([np.nanmin(dataset, axis = 0),
                      temp_mean,
                      np.nanmax(dataset, axis = 0)])


# In[8]:


temp_stats


# # Now we need to seperate num and str columns

# #### This is list of text columns

# In[9]:


str_column = np.argwhere(np.isnan(temp_mean)).squeeze()
str_column


# #### This is list of numeric columns

# In[10]:


num_column = np.argwhere(np.isnan(temp_mean) == False).squeeze()
num_column


# # Now we will reimport the data

# #### This is string dataset

# In[11]:


dataset_str = np.genfromtxt("loan-data.csv", delimiter = ';', skip_header = 1, 
                            autostrip = True, usecols = str_column, dtype = str)
dataset_str


# #### This is numeric dataset

# In[12]:


dataset_num = np.genfromtxt("loan-data.csv", delimiter = ';', skip_header = 1, 
                            autostrip = True, usecols = num_column, filling_values = temp_fill)
dataset_num


# # Header

# In[13]:


Header_full = np.genfromtxt("loan-data.csv", delimiter = ';',autostrip = True, skip_footer = dataset.shape[0],dtype = str)
Header_full


# In[14]:


header_strings, header_numeric = Header_full[str_column], Header_full[num_column]


# In[15]:


header_strings


# In[16]:


header_numeric


# In[17]:


header_strings[0] = "issue_date"


# In[18]:


header_strings


# # Issue_Date column

# In[19]:


np.unique(dataset_str[:,0])


# In[20]:


dataset_str[:,0] = np.chararray.strip(dataset_str[:,0], "-15")


# In[21]:


np.unique(dataset_str[:,0])


# In[22]:


months = np.array(['','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])


# In[23]:


months


# #### It iterates over each index (from 0 to 12) in the range of 13.
# #### For each index 'i', it checks if the element in the first column of 'dataset_str' matches the abbreviated month name at index 'i' in the 'months' array.
# #### If there is a match, it replaces that element with 'i', which is the numerical representation of the month.
# #### If there is no match, it leaves the element unchanged.

# In[24]:


for i in range(13):
    dataset_str[:,0] = np.where(dataset_str[:,0] == months[i],
                               i,
                               dataset_str[:,0])


# In[25]:


np.unique(dataset_str[:,0])


# In[26]:


dataset_str[:,0]


# # Loan status column

# In[27]:


dataset_str[:,1]


# In[28]:


np.unique(dataset_str[:,1])


# #### The column loan status have 8 unique values and blank values.
# #### We can put this status into two categories namely Good and Bad.
# #### The good category includes, 'current', 'Fully Paid', 'In Grace Period', 'Issued' and 'Late (16-30 days)'. While the bad constists of the leftovers.

# In[30]:


status_bad = np.array(['', 'Charged Off', 'Default', 'Late (31-120 days)'])


# In[31]:


status_bad


# In[32]:


dataset_str[:,1]


# In[33]:


dataset_str[:,1] = np.where(np.isin(dataset_str[:,1], status_bad),0,1)


# In[35]:


np.unique(dataset_str[:,1])


# ## Term column

# In[37]:


np.unique (dataset_str[:,2])


# In[38]:


dataset_str[:,2] = np.chararray.strip(dataset_str[:,2], " months")
dataset_str[:,2]


# In[39]:


header_strings[2] = "Term months"


# In[40]:


header_strings


# In[41]:


np.unique(dataset_str[:,2])


# #### We can see that we have missing data. We can fill this with 60(considering the worst scenerio) as this dataset is for Credit Risk modelling.

# In[42]:


dataset_str[:,2] = np.where(dataset_str[:,2] == '',
                            '60',
                            dataset_str[:,2])


# In[43]:


np.unique(dataset_str[:,2])


# #### Column grade and subgrade

# In[44]:


np.unique(dataset_str[:,3])


# ##### we have 7 unique category in grade column

# In[45]:


np.unique(dataset_str[:,4])


# ##### we have 36 unique values in sub grade

# ### Now we will fill the missing values in subgrade column

# In[47]:


for i in np.unique(dataset_str[:,3])[1:]:
    dataset_str[:,4] = np.where((dataset_str[:,4] == '') & (dataset_str[:,3] == i), i + '5',
                                          dataset_str[:,4])


# In[52]:


np.unique(dataset_str[:,4], return_counts = True)


# #### still there are 9 missing vaues in the sub grade column. We can delete these row but instead of deleting, we can assign them the lowest grade as these data is for CRM

# In[53]:


dataset_str[:,4] = np.where(dataset_str[:,4] == '',
                             'G5',
                             dataset_str[:,4])


# In[54]:


np.unique(dataset_str[:,4], return_counts = True)


# #### The grade can be known from subgrade so, we can delete the column "grade"

# In[58]:


dataset_str = np.delete(dataset_str, 3, axis = 1)


# In[59]:


dataset_str[:,3]


# In[60]:


header_strings = np.delete(header_strings, 3)


# In[61]:


header_strings


# ## Column Verification Status

# In[62]:


header_strings


# In[63]:


np.unique(dataset_str[:,4])


# In[ ]:


#### The blank and 'Not verified' status are replaced with 0 while the 'Verified' and 'Source Verified' are replaced with 1


# In[64]:


dataset_str[:,4] = np.where((dataset_str[:,4] == '') | (dataset_str[:,4] == 'Not Verified'),0,1)


# In[65]:


np.unique(dataset_str[:,4])


# ## Column URL

# In[66]:


dataset_str[:,5]


# In[67]:


np.chararray.strip(dataset_str[:,5], "https://www.lendingclub.com/browse/loanDetail.action?loan_id=")


# In[69]:


dataset_num[:,0]


# #### It looks like the loan id we got from url is same as id number of th customer.

# In[74]:


dataset_str[:,5] = np.chararray.strip(dataset_str[:,5], "https://www.lendingclub.com/browse/loanDetail.action?loan_id=")


# In[75]:


np.array_equal(dataset_num[:,0].astype(dtype = np.int32), dataset_str[:,5].astype(dtype = np.int32))


# #### they are same, so, we can delete this column as it's of no use in analysis

# In[76]:


dataset_str = np.delete(dataset_str,5, axis = 1)
header_strings = np.delete(header_strings,5)


# In[77]:


header_strings


# In[79]:


dataset_num[:,0]


# # Column "addr_state"

# In[80]:


header_strings


# In[81]:


header_strings[5] = "State Address"


# In[82]:


header_strings


# In[86]:


np.unique(dataset_str[:,5], return_counts = True)


# In[85]:


np.unique(dataset_str[:,5]).size


# #### we have 49 unique states and 500 missing values.
# #### THE US have 50 states out of which Iowa is missing. One possibility could be it was picked as benchmark.

# #### we can fill the missing value with Iowa (IA)

# In[87]:


dataset_str[:,5] = np.where(dataset_str[:,5] == '',
                             'IA',
                             dataset_str[:,5])


# In[88]:


np.unique(dataset_str[:,5], return_counts = True)


# #### Now we have done cleaning and processing the text data we can create a checkpoint

# In[92]:


def checkpoint(file_name, checkpoint_header, checkpoint_data):
    np.savez(file_name, header = checkpoint_header, data = checkpoint_data)
    checkpoint_variable = np.load(file_name + ".npz")
    return(checkpoint_variable)


# In[93]:


checkpoint_strings = checkpoint("checkpoint-strings", header_strings, dataset_str)


# In[94]:


checkpoint_strings["header"]


# In[95]:


checkpoint_strings["data"]


# #### just to ensure the data have been same

# In[96]:


np.array_equal(checkpoint_strings['data'], dataset_str)


# In[97]:


np.array_equal(checkpoint_strings['header'], header_strings)


# # Now we will clean and preprocess the numeric columns

# In[98]:


dataset_num


# In[100]:


header_numeric


# #### we will check if there is any missing data

# In[101]:


np.sum(np.isnan(dataset_num))


# In[116]:


temp_stats[:, num_column]


# In[104]:


temp_stats


# ## Earlier we temperory filled the numeric data with max+1 and imported the dataset, now we will replace all those temporyry fills with the actual values.

# ### column id

# In[108]:


np.isin(dataset_num[:,0],temp_fill).sum()


# In[110]:


##In the 1st column we don't have any temporary filled values. So, no changes necessary.


# ### Column Funded amount

# In[113]:


### For this column we will replace the temp fill with the minimum value of this column


# In[114]:


dataset_num[:,2]


# In[117]:


dataset_num[:,2] = np.where(dataset_num[:,2] == temp_fill,
                           temp_stats[0, num_column[2]],
                           dataset_num[:,2])


# In[118]:


dataset_num[:,2]


# ### Column loaned Amount, Intrest rate, Total payment, Installment

# In[120]:


## we will replace the temp values with the maximum value of each column respectively.


# In[121]:


for i in [1,3,4,5]:
    dataset_num[:,i] = np.where(dataset_num[:,i] == temp_fill,
                           temp_stats[2, num_column[i]],
                           dataset_num[:,i])


# In[122]:


dataset_num


# ### Completing the datasheet

# In[123]:


dataset_str.shape


# In[124]:


dataset_num.shape


# In[127]:


np.hstack((dataset_num, dataset_str)).shape


# In[128]:


final_data = np.hstack((dataset_num, dataset_str))


# In[129]:


final_data


# In[137]:


final_header = np.concatenate((header_numeric, header_strings))


# In[138]:


final_header


# ## combining final header and final data

# In[139]:


final_loan_datasheet = np.vstack((final_header, final_data))


# In[140]:


final_loan_datasheet


# In[141]:


np.savetxt("final_loan_data.csv",
           final_loan_datasheet,
          fmt = "%s",
          delimiter = ',')


# In[ ]:




