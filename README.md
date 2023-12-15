# Job-Remote-Classification
This project is focused on classifying jobs as remote or non-remote based on job descriptions using BERT (Bidirectional Encoder Representations from Transformers). It demonstrates an accuracy of approximately **96%** for descriptions exceeding 100 words. It can classify ~500 descriptions per minute even on a small GPU. The primary purpose of this project is to process data for the working paper titled "Labor Mobility and Productivity Effects of Pandemic-Induced Work-From-Home Policies." 

I uploaded the python file **"BERT_JOB_CLASSIFICATION_TRAIN.py"** which trains the BERT model and saves it. **"PREDICT_AND_TEST.ipynb"** provides the code for loading the model to predict on a test set. At last, the model I trained is here:  (For Replication: Create a new folder and put the model.bin and config.json in this repository into the folder; Then, run the Jupyter Notebook code with path directed to the folder)

**Note**: This approach is much more efficient and faster than other methods I tested (ChatGPT, Llama 2, etc.). It also does not require a very strong GPU such as the A100s. 

Yongzhe Wang (Andrew)
