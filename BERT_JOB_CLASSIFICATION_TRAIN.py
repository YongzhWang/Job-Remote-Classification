import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tabulate import tabulate
from tqdm import trange
import random
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re

nltk.download("stopwords")

# remove stopwords
def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    words = text.split()  
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return " ".join(filtered_words) 

#Read Data and pre-process train data
df=pd.read_csv("NEW_split_train.csv")

output_1_rows = df[df['output1'] == 1]
output_0_rows = df[df['output1'] == 0]
print(len(output_1_rows),len(output_0_rows))

#Take random 25000 observations remote and non-remote job each as training data
sample_size = 25000  
half_sample_size = sample_size

# Concatenate the two samples
sample_df = pd.concat([output_1_rows[0:half_sample_size], output_0_rows[0:half_sample_size+6000]])
test_df = pd.concat([output_1_rows[half_sample_size:], output_0_rows[half_sample_size+6000:]])
test_df.to_csv("test_set.csv")


# Use regular expression to remove non-space, non-letter, non-number, non-comma, non-period, and non-question mark characters
def preprocess_string(input_string):
    cleaned_string = re.sub(r'[^a-zA-Z0-9\s,.\?]', '', input_string)
    return cleaned_string

df['input1'] = df['input1'].apply(preprocess_string)
df['input1'] = df['input1'].apply(remove_stopwords)

text = df.input1.values
labels = df.output1.values


#Start Training with BERT
tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased',
    do_lower_case = True
    )


def print_rand_sentence():
  '''Displays the tokens and respective IDs of a random text sample'''
  index = random.randint(0, len(text)-1)
  table = np.array([tokenizer.tokenize(text[index]), 
                    tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text[index]))]).T
  print(tabulate(table,
                 headers = ['Tokens', 'Token IDs'],
                 tablefmt = 'fancy_grid'))

#Take a look at a random sentence
print_rand_sentence()

token_id = []
attention_masks = []

def preprocessing(input_text, tokenizer):
  '''
    - input_ids: list of token ids
    - token_type_ids: list of token type ids
    - attention_mask: list of indices (0,1) specifying which tokens should considered by the model (return_attention_mask = True).
  '''
  return tokenizer.encode_plus(
                        input_text,
                        add_special_tokens = True,
                        max_length = 512,
                        pad_to_max_length = True,
                        return_attention_mask = True,
                        return_tensors = 'pt'
                   )


for sample in text:
  encoding_dict = preprocessing(sample, tokenizer)
  token_id.append(encoding_dict['input_ids']) 
  attention_masks.append(encoding_dict['attention_mask'])

token_id = torch.cat(token_id, dim = 0)
attention_masks = torch.cat(attention_masks, dim = 0)
labels = torch.tensor(labels)


def print_rand_sentence_encoding():
  '''Displays tokens, token IDs and attention mask of a random text sample'''
  index = random.randint(0, len(text) - 1)
  tokens = tokenizer.tokenize(tokenizer.decode(token_id[index]))
  token_ids = [i.numpy() for i in token_id[index]]
  attention = [i.numpy() for i in attention_masks[index]]

  table = np.array([tokens, token_ids, attention]).T
  print(tabulate(table, 
                 headers = ['Tokens', 'Token IDs', 'Attention Mask'],
                 tablefmt = 'fancy_grid'))

print_rand_sentence_encoding()



val_ratio = 0.1
# Reference: https://arxiv.org/pdf/1810.04805.pdf
batch_size = 16

# Indices of the train and validation splits stratified by labels
train_idx, val_idx = train_test_split(
    np.arange(len(labels)),
    test_size = val_ratio,
    shuffle = True,
    stratify = labels)

# Train and validation sets
train_set = TensorDataset(token_id[train_idx], 
                          attention_masks[train_idx], 
                          labels[train_idx])

val_set = TensorDataset(token_id[val_idx], 
                        attention_masks[val_idx], 
                        labels[val_idx])

# Prepare DataLoader
train_dataloader = DataLoader(
            train_set,
            sampler = RandomSampler(train_set),
            batch_size = batch_size
        )

validation_dataloader = DataLoader(
            val_set,
            sampler = SequentialSampler(val_set),
            batch_size = batch_size
        )


#True Positives (TP)
def b_tp(preds, labels):
    return sum([preds == labels and preds == 1 for preds, labels in zip(preds, labels)])
#False Positives (FP)
def b_fp(preds, labels):
    return sum([preds != labels and preds == 1 for preds, labels in zip(preds, labels)])
#True Negatives (TN)
def b_tn(preds, labels):
    return sum([preds == labels and preds == 0 for preds, labels in zip(preds, labels)])
#False Negatives (FN)
def b_fn(preds, labels):
    return sum([preds != labels and preds == 0 for preds, labels in zip(preds, labels)])

    # accuracy    = (TP + TN) / N
    # precision   = TP / (TP + FP)
    # recall      = TP / (TP + FN)
    # specificity = TN / (TN + FP)
def b_metrics(preds, labels):
    preds = np.argmax(preds, axis = 1).flatten()
    labels = labels.flatten()
    tp = b_tp(preds, labels)
    tn = b_tn(preds, labels)
    fp = b_fp(preds, labels)
    fn = b_fn(preds, labels)
    b_accuracy = (tp + tn) / len(labels)
    b_precision = tp / (tp + fp) if (tp + fp) > 0 else 'nan'
    b_recall = tp / (tp + fn) if (tp + fn) > 0 else 'nan'
    b_specificity = tn / (tn + fp) if (tn + fp) > 0 else 'nan'
    return b_accuracy, b_precision, b_recall, b_specificity










# Load the BertForSequenceClassification model
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels = 2,
    output_attentions = False,
    output_hidden_states = False,
)

# Reference for learning rate: https://arxiv.org/pdf/1810.04805.pdf
optimizer = torch.optim.AdamW(model.parameters(), 
                              lr = 5e-5,
                              eps = 1e-08
                              )

# Run on GPU
model.cuda()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Reference: https://arxiv.org/pdf/1810.04805.pdf
epochs = 2

#Training
for _ in trange(epochs, desc = 'Epoch'):
    model.train()
    
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0

    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        optimizer.zero_grad()
        train_output = model(b_input_ids, 
                             token_type_ids = None, 
                             attention_mask = b_input_mask, 
                             labels = b_labels)
        train_output.loss.backward()
        optimizer.step()
        tr_loss += train_output.loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1

    # Validation
    model.eval()

    # Tracking variables 
    val_accuracy = []
    val_precision = []
    val_recall = []
    val_specificity = []

    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
          # Forward pass
          eval_output = model(b_input_ids, 
                              token_type_ids = None, 
                              attention_mask = b_input_mask)
        logits = eval_output.logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        # Calculate validation metrics
        b_accuracy, b_precision, b_recall, b_specificity = b_metrics(logits, label_ids)
        val_accuracy.append(b_accuracy)
        # Update precision only when (tp + fp) !=0; ignore nan
        if b_precision != 'nan': val_precision.append(b_precision)
        # Update recall only when (tp + fn) !=0; ignore nan
        if b_recall != 'nan': val_recall.append(b_recall)
        # Update specificity only when (tn + fp) !=0; ignore nan
        if b_specificity != 'nan': val_specificity.append(b_specificity)

    print('\n\t - Train loss: {:.4f}'.format(tr_loss / nb_tr_steps))
    print('\t - Validation Accuracy: {:.4f}'.format(sum(val_accuracy)/len(val_accuracy)))
    print('\t - Validation Precision: {:.4f}'.format(sum(val_precision)/len(val_precision)) if len(val_precision)>0 else '\t - Validation Precision: NaN')
    print('\t - Validation Recall: {:.4f}'.format(sum(val_recall)/len(val_recall)) if len(val_recall)>0 else '\t - Validation Recall: NaN')
    print('\t - Validation Specificity: {:.4f}\n'.format(sum(val_specificity)/len(val_specificity)) if len(val_specificity)>0 else '\t - Validation Specificity: NaN')
    
    

    
#Check a sample    
new_sentence = '''
Part Time Assistant Manager 8055 W Bowles Ave  Store 0013  Tuesday Morning  Littleton, CO Tuesday Morning    Job Company  Job details  Salary 14.35  23.00 an hour Job Type Parttime  Full Job Description  Tuesday Morning is taking the lead in offprice retail offering upscale decorative home accessories, housewares, seasonal goods and famousmaker gifts.  Our mission is simple offer fresh and exciting merchandise at unbelievable value, with impeccable service.  With over 750 stores in 40 states, and continuing to grow, we are always seeking strong leadership to fuel our growth.  The Part Time Assistant Store Managers role is to, take the lead from and, partner with the Store Manager to engage, motivate and lead a team of associates in operating a profitable store, while creating a positive environment for the associate and the guest. The Assistant Store Manager is the extension of the Store Manager and will provide overall support to drive the Store Managers vision and direction for the store.  Responsibilities  Sales Driving sales by creating a sales generating environment through the implementation of all corporate sales directives. Service Foster a service oriented environment tailored to the unique seeker, and ensuring the guest is always taken care of the right way. Merchandise Ensure Merchandising standards and product presentations are second to none, and create that WOW factor. Leadership Provide ongoing coaching feedback, empowering your team to do whats right, setting clear expectations and leading by example. Communication Set the vision and direction for the store, share information to align your team  help them feel a part of something big.  Skills  experience  23 years of progressively responsible retail, and at least 1 year of supervision, experience required. Must understand and be able to execute concepts related to financial principles, inventory management, and merchandising. Bachelors degree preferred. Possess strong leadership skills with the ability to train, coach and mentor associates with professional maturity. Ability to make decisions, communicate, analyze financial information, problem solve, organization and computer skills. Must be 21 years of age. Ability to relocate, for future growth and promotional opportunities, strongly desired.  We offer competitive compensation, excellent benefits to include 401k, bestinclass products and more, in a high performing environment. Working in our stores provides you with unlimited possibilities to start or expand your career.  Pay Range 14.35  23.00hr  Benefits  Join Tuesday Morning and enjoy  Some of the best hours in retail 401K 20 Associate discount Rewarding career with advancement opportunities  CB  Tuesday Morning 
'''

test_ids = []
test_attention_mask = []

# Apply the tokenizer
encoding = preprocessing(new_sentence, tokenizer)

# Extract IDs and Attention Mask
test_ids.append(encoding['input_ids'])
test_attention_mask.append(encoding['attention_mask'])
test_ids = torch.cat(test_ids, dim = 0)
test_attention_mask = torch.cat(test_attention_mask, dim = 0)

with torch.no_grad():
    output = model(test_ids.to(device), token_type_ids = None, attention_mask = test_attention_mask.to(device))

prediction = 'Remote' if np.argmax(output.logits.cpu().numpy()).flatten().item() == 1 else 'Non-Remote'
# Save model for future use
model.save_pretrained("001Model", from_pt=True) 

print('Input Sentence: ', new_sentence)
print('Predicted Class: ', prediction)