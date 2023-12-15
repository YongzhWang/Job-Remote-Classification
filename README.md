# Job-Remote-Classification
This project is focused on classifying jobs as remote or non-remote based on job descriptions using BERT (Bidirectional Encoder Representations from Transformers). It demonstrates an accuracy of approximately **96%** for descriptions exceeding 100 words. It can classify **~500 descriptions per minute** even on a small GPU. The primary purpose of this project is to process and classify the 13 million Job Posting data for the working paper titled "Labor Mobility and Productivity Effects of Pandemic-Induced Work-From-Home Policies." 

I uploaded the python file **"BERT_JOB_CLASSIFICATION_TRAIN.py"** which trains the BERT model and saves it. **"PREDICT_AND_TEST.ipynb"** provides the code for loading the model to predict on a test set. Last, the model I trained is here: https://drive.google.com/file/d/10Q4d_u6HnOrs0Ss8X6NpQph46NAebpKL/view?usp=sharing (For Replication: Create a new folder and put the model.bin and config.json in this repository into the folder; Then, run the Jupyter Notebook code with path directed to the folder.) For people interested in the distribution of the data, some preliminary analysis is done in **"SUM_STATE_DIST_CODE.py"** and a few output figures and tables on the distribution across state and time are also included. The distribution over time:

![alt text]([https://github.com/Cat-Like-IceCream/Job-Remote-Classification/[Distribution]SUM_ALL_DATA_JOB_POSTING.jpg?raw=true](https://github.com/Cat-Like-IceCream/Job-Remote-Classification/blob/main/%5BDistribution%5DSUM_ALL_DATA_JOB_POSTING.jpg))

**Note**: This approach is much more efficient and faster than other methods I tested (ChatGPT, Llama 2, etc.). It also does not require a very strong GPU such as the A100s. 

Yongzhe Wang (Andrew)
