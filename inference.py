from modules.utils import Config
import argparse

from dataloader.hansoldataset import load_hansol_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

from peft import PeftConfig, PeftModel
from tqdm import tqdm

import pandas as pd
import re
from sentence_transformers import SentenceTransformer 
print('hi')
def main(CFG):
    
    # Load LORA MODEL
    print('load : ', CFG.new_model)
    config = PeftConfig.from_pretrained(CFG.new_model)
    base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(base_model, CFG.new_model)
    model.to(CFG.device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" 

    print("=" * 80)
    print("model is in device : " + str(model.device))
    print("=" * 80)


    test = pd.read_csv('./data/test.csv')
    test['질문'] = test['질문'].apply(lambda x : re.split('[?!.]', x))

    preds = []
    for test_questions in tqdm(test['질문']):
        # 각 질문 row 별로 대답 저장
        preds_temp = []

        for test_question in test_questions:
            # ?!. 으로 split했을 때 공백이 나눠지는 경우 제외
            if len(test_question) == 0: 
                continue
            # 입력 텍스트를 토큰화하고 모델 입력 형태로 변환
            input_ids = tokenizer.encode('질문 : ' + test_question + '답변 : ', return_tensors='pt')

            # 답변 생성
            output_sequences = model.generate(
                input_ids=input_ids.to(CFG.device),
                max_length=500,
                temperature=0.9,
                top_k=1,
                top_p=0.9,
                repetition_penalty=1.3,
                do_sample=True,
                num_return_sequences=1
            )

            # 생성된 텍스트(답변) 저장
            for generated_sequence in output_sequences:
                full_text = tokenizer.decode(generated_sequence, skip_special_tokens=False)
                # 질문과 답변의 사이를 나타내는 '답변 :'을 찾아, 이후부터 출력 해야 함
                answer_start = full_text.find(tokenizer.eos_token) + len(tokenizer.eos_token)
                answer_only = full_text[answer_start:].strip()
                answer_only = answer_only.replace('\n', ' ')
                preds_temp.append(answer_only)
            
        print(preds_temp)
        preds.append(preds_temp)


    # preds 후처리
    new_preds = []
    for pred in preds:
        temp = ''
        for sentence in pred:
            sentence = sentence.replace('\n', ' ').replace('</s>', ' ')
            temp += sentence
        temp += '</s>'
        new_preds.append(temp)


    # Embedding Vector 추출에 활용할 모델(distiluse-base-multilingual-cased-v1) 불러오기
    model_sentence = SentenceTransformer('distiluse-base-multilingual-cased-v1')

    # 생성한 모든 응답(답변)으로부터 Embedding Vector 추출
    pred_embeddings = model_sentence.encode(new_preds)

    submit = pd.read_csv('./data/sample_submission.csv')
    # 제출 양식 파일(sample_submission.csv)을 활용하여 Embedding Vector로 변환한 결과를 삽입
    result_df = pd.DataFrame()
    result_df['id'] = submit['id']
    result_df['result'] = new_preds
    result_df.to_csv(f'./submission/{CFG.new_model}_result.csv', index=False)

    submit.iloc[:,1:] = pred_embeddings
    submit.to_csv(f'./submission/{CFG.new_model}_embedding.csv', index=False)

        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=None)

    args = parser.parse_args()

    CFG = Config()
    CFG = CFG.from_yaml('./configs/config.yaml')
    CFG_custom = Config()
    CFG_custom = CFG.from_yaml('./configs/' + args.config)
    CFG.update(CFG_custom)
    CFG.device = 'cuda:' + str(args.gpu)

    main(CFG)