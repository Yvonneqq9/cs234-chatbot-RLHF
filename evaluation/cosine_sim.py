#%%
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import torch
#%% load data
data=pd.read_csv('merged_output.csv')

#%% load model and tokenizer
model_path='gpt2'
tokenizer=AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
tokenizer.truncation_side = 'left'

#%%
data['answer0_token']=np.nan
data['answer1_token']=np.nan
data['reference_token']=np.nan

with torch.no_grad():
    for i in range(len(data['answer_0'])):
        data['answer0_token'][i] = tokenizer(data['answer_0'][i],truncation=True, padding='max_length',max_length=512, return_tensors="pt")
        data['answer1_token'][i] = tokenizer(data['answer_1'][i], truncation=True, padding='max_length',max_length=512,return_tensors="pt")
        data['reference_token'][i] = tokenizer(data['reference_answer'][i], truncation=True, padding='max_length',max_length=512,return_tensors="pt")
# return a dictionary of 'input_ids' and 'attention_mask'
#%%
import torch.nn.functional as F
def cos(input1,input2):
    assert input1.shape==input2.shape
    input1 = torch.tensor(input1, dtype=torch.float32)
    input2 = torch.tensor(input2, dtype=torch.float32)
    result=F.cosine_similarity(input1,input2,dim=-1)
    return result

#%%
data=data.reset_index(drop=True)
data['cosine_ans0']=np.nan
data['cosine_ans1']=np.nan
for i in range(len(data)):
    data['cosine_ans0'][i]=cos(torch.tensor(data['answer0_token'][i]['input_ids'], dtype=torch.float32),torch.tensor(data['reference_token'][i]['input_ids'],dtype=torch.float32))
    data['cosine_ans1'][i] = cos(torch.tensor(data['answer1_token'][i]['input_ids'], dtype=torch.float32),torch.tensor(data['reference_token'][i]['input_ids'],dtype=torch.float32))
#%%
answer0_result=np.average(data['cosine_ans0'])
answer1_result=np.average(data['cosine_ans1'])
#%%
print(answer0_result,answer1_result)