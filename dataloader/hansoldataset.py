import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def load_dataset(tokenizer, split = None):
    if split == 'train':
        return TrainDataset(tokenizer)
    elif split == 'test':
        return TestDataset(tokenizer)
    else:
        raise("no split detected")

class TrainDataset(Dataset):
    def __init__(self, tokenizer, max_len = 300):
        self.tokenizer = tokenizer
        self.data = []
        train_df = pd.read_csv('./data/train.csv')
        # Iterate through the DataFrame and format data
        for _, row in tqdm(train_df.iterrows(), desc="Formatting data"):
            for q_col in ['질문_1', '질문_2']:
                for a_col in ['답변_1', '답변_2', '답변_3', '답변_4', '답변_5']:
                    # 질문과 답변 쌍을 </s> token으로 연결
                    input_text = row[q_col] + self.tokenizer.eos_token + row[a_col]
                    # Remove batch dimension
                    input_ids = self.tokenizer.encode(input_text, return_tensors='pt').squeeze(0) 
                    self.data.append(input_ids)

        truncated_data_tensors = [seq[:max_len] for seq in self.data]
        # Then pad the truncated sequences
        self.data = pad_sequence(truncated_data_tensors, batch_first=True, padding_value=tokenizer.eos_token_id)
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

    def __max_len__(self):
        max_len =  max([len(sen) for sen in self.data])
        return max_len
    

class TestDataset(Dataset):
    def __init__(self, tokenizer, max_len = 300):
        self.tokenizer = tokenizer
        self.data = []
        train_df = pd.read_csv('./data/test.csv')
        # Iterate through the DataFrame and format data
        for test_question in tqdm(train_df['질문']):
            input_ids = tokenizer.encode('질문 : ' + test_question + '답변 : ' , return_tensors='pt')
            self.data.append(input_ids)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

