import pandas as pd
from itertools import product
from sklearn.model_selection import train_test_split

train = pd.read_csv("./train.csv")

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

train.to_csv("./train_data_gemma.csv",index=False,encoding='utf-8')
valid.to_csv("./valid_data_gemma.csv",index=False,encoding='utf-8')


