from sentence_transformers import SentenceTransformer 
import pandas as pd


def concat_answer(answer_df : pd.DataFrame) -> pd.DataFrame:
    # Ensure the 'answer' column is a string if it's not already
    answer_df['answer'] = answer_df['answer'].apply(lambda x: ''.join(x))
    # Group by 'id' and concatenate the answers
    answer_by_id_df = answer_df.groupby('id')['answer'].apply(''.join).reset_index()

    # Embedding Vector 추출에 활용할 모델(distiluse-base-multilingual-cased-v1) 불러오기
    model_sentence = SentenceTransformer('distiluse-base-multilingual-cased-v1')

    pred_embeddings = model_sentence.encode(answer_by_id_df['answer'])

    submission_df = pd.read_csv('../data/sample_submission.csv')
    # 제출 양식 파일(sample_submission.csv)을 활용하여 Embedding Vector로 변환한 결과를 삽입
    submission_df.iloc[:,1:] = pred_embeddings
    return submission_df

if __name__ == '__main__':
    # useage example. change answer_file to test.
    answer_file = 'SOLAR_lora256_rag_True_ft_True'
    answer_df = pd.read_csv(f'../submission/{answer_file}.csv', encoding = 'utf-8')
    submission_df = concat_answer(answer_df)
    submission_df.to_csv(f'../submission/{answer_file}_embedding.csv', index=False)
