from sentence_transformers import SentenceTransformer 
import pandas as pd

### 미완성 파일

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
