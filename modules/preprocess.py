import pandas as pd
from itertools import product
from sklearn.model_selection import train_test_split
import re

datapath = '../data'

train = pd.read_csv(datapath + "/train.csv")

train_data = []

for q,a in list(product([f"질문_{x}" for x in range(1,3)],[f"답변_{x}" for x in range(1,6)])):
    for i in range(len(train)):
        train_data.append(
            "<start_of_turn>user 다음 질문에 대해 정확하게 답하시오. 질문: "+ train.at[i,q] + " 답변 : <end_of_turn><start_of_turn>model" + train.at[i,a]
        )

data = pd.DataFrame(train_data)
train, valid = train_test_split(data, test_size=0.05, random_state=42)
train.columns = ['QnA']
valid.columns = ['QnA']

train.to_csv(datapath + "/train_data_gemma.csv",index=False,encoding='utf-8')
valid.to_csv(datapath + "/valid_data_gemma.csv",index=False,encoding='utf-8')

train = pd.read_csv(datapath + "/train.csv")

train_data = []

for q,a in list(product([f"질문_{x}" for x in range(1,3)],[f"답변_{x}" for x in range(1,6)])):
    for i in range(len(train)):
        train_data.append(
            "<start_of_turn>user 다음 질문에 대해 정확하게 답하시오. 질문: "+ train.at[i,q] + " 답변 : <end_of_turn><start_of_turn>model" + train.at[i,a]
        )

data = pd.DataFrame(train_data)
train, valid = train_test_split(data, test_size=0.05, random_state=42)
train.columns = ['QnA']
valid.columns = ['QnA']

train.to_csv(datapath + "/train_data_solar.csv",index=False,encoding='utf-8')
valid.to_csv(datapath + "/valid_data_solar.csv",index=False,encoding='utf-8')



# Test set preprocessing
test = pd.read_csv(datapath + "/test.csv")

test['질문'] = test['질문'].apply(lambda x : re.split('[?!.]', x))
preds_temp = []

for i in range(len(test)):
    # 각 질문 row 별로 대답 저장
    for test_question in test.at[i,'질문']:
        # ?!. 으로 split했을 때 5개 이하의 글자로 나눠지는 경우 제외
        if len(test_question) < 5: 
            continue
        test_question= test_question.replace('또한, ','').replace('그리고', '').replace('"','').lstrip()
        test_question += '?'
        preds_temp.append({'id' : test.at[i,'id'], 'Question' : test_question})


test_cleaned = pd.DataFrame(preds_temp)
test_cleaned.to_csv(datapath + '/test_cleaned.csv', index=False)
