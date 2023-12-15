import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Reading a text file, splitting each line by comma, and creating a DataFrame
with open('NEWOUTPUT.txt', 'r') as file:
    lines = file.readlines()
data = [line.strip().split(',') for line in lines]
df = pd.DataFrame(data, columns=['job_id', 'output_BERT'])

# Converting 'job_id' column to integer type
df['job_id'] = df['job_id'].astype(int) 

# Function to read multiple CSV files in a specified range and concatenate them into a single DataFrame
def read_csv_files_in_range(folder_path, start_index, end_index):
    data_frames = pd.DataFrame()
    for i in range(start_index, end_index + 1):
        file_name = f"{i:03d}.part"
        file_path = os.path.join(folder_path, file_name)
        print(file_path)
        df = pd.read_csv(file_path)
        data_frames=pd.concat([data_frames,df])

    return data_frames

# Using the function to read and merge CSV files in a given folder path and range
folder_path = "Data/part"
df_merge = read_csv_files_in_range(folder_path, 0, 140)
df_merge=df_merge.rename({"Unnamed: 0":"ID"},axis=1)

# Reading a CSV file, naming the columns, and performing some transformations
column_names = ["id", "col1", "col2","col3"]
df_exclude = pd.read_csv("Code/output.txt", names=column_names)

# Calculating the maximum value across three columns and creating a binary variable based on a threshold
max_value = df_exclude[['col1', 'col2', 'col3']].max(axis=1)
threshold = 0.4
binary_variable = (max_value > threshold).astype(int)
df_exclude['output_BERT'] = binary_variable

# Dropping unnecessary columns and renaming a column for consistency
df_exclude.drop(columns=['col1', 'col2', 'col3'], inplace=True)
df_exclude.rename(columns={'id': 'job_id'}, inplace=True)
df = df.astype(int)
df_exclude = df_exclude.astype(int)

# Concatenating two DataFrames and merging with another DataFrame based on 'job_id'
df = pd.concat([df_exclude, df], axis=0).reset_index(drop=True)
df = pd.merge(df_merge, df, on='job_id', how='inner')

# Save for later use
df.to_csv("FINAL.csv",index=False)
print(len(df))

# Converting 'post_date' to datetime format and extracting year and month into separate columns
df['post_date'] = pd.to_datetime(df['post_date'])
df['year'] = df['post_date'].dt.year
df['month'] = df['post_date'].dt.month
df["output_BERT"]=1

# Grouping data by state, year, and month and summing up 'output_BERT'
proportions = df.groupby(['state', 'year', 'month'])['output_BERT'].sum().reset_index()

# Creating a pivot table with 'state' as rows and 'year' and 'month' as columns
pivot_table = proportions.pivot_table(index='state', columns=['year', 'month'], values='output_BERT')
pivot_table = pivot_table.fillna(0)

# Saving the pivot table
pivot_table.to_csv("SUM_STATE_JOB_POSTING_DISTR.csv")

# Creating a 'year-month' column for plotting
df['year-month'] = df['year'].astype(str) + '-' + df['month'].apply(lambda x: str(x).zfill(2)
proportions = df.groupby('year-month')['output_BERT'].sum().reset_index()

# Plotting the data as a line graph
plt.figure(figsize=(15, 12))
plt.plot(proportions['year-month'].to_numpy(), proportions['output_BERT'].to_numpy(), marker='o', linestyle='-')
plt.xlabel('Year-Month')
plt.ylabel('Job_Posting Remote Proportion')
plt.title('Job_Posting Remote Proportion across All States')
plt.xticks(rotation=45)
plt.grid(True)
plt.gca().set_facecolor('white')  # Set the background color to white

# Saving the plot as an image and displaying it
plt.savefig('SUM_ALL_DATA_JOB_POSTING_DISTR.jpg', bbox_inches='tight')
plt.show()

                                                                    
                                                                    
                                                                    
df = pd.read_csv("FINAL.csv")

# Converting 'post_date' to datetime and filtering for dates from 2020 onwards
df['post_date'] = pd.to_datetime(df['post_date'])
df = df[df['post_date'] >= '2020-01-01']

# Grouping by entity and company, calculating total and remote job postings, and computing their proportion
result = df.groupby(['factset_entity_id', "company"]).agg(
    total_postings=('output_BERT', 'count'),
    remote_postings=('output_BERT', 'sum')
)
result['proportion_remote'] = result['remote_postings'] / result['total_postings']

# Resetting the index and renaming columns for clarity, then saving the result to a CSV file
result.reset_index(inplace=True)
result = result.rename(columns={'factset_entity_id': 'Firm', 'total_postings': 'Total Postings', 'remote_postings': 'Remote Postings'})
result.to_csv("FIRM_FINAL.csv", index=False)

# Similar process as above but grouping also by 'state', followed by saving the result to another CSV file
result = df.groupby(['factset_entity_id', 'state', "company"]).agg(
    total_postings=('output_BERT', 'count'),
    remote_postings=('output_BERT', 'sum')
)
result['proportion_remote'] = result['remote_postings'] / result['total_postings']
result.reset_index(inplace=True)
result = result.rename(columns={'factset_entity_id': 'Firm', 'state': 'State', 'total_postings': 'Total Postings', 'remote_postings': 'Remote Postings'})

#Save
result.to_csv("FIRM_STATE_FINAL.csv", index=False)
