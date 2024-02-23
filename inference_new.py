from modules.utils import Config
import argparse

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

from peft import PeftConfig, PeftModel
from tqdm import tqdm

import pandas as pd
import re
from sentence_transformers import SentenceTransformer 

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.chroma import Chroma
# from data_loader import load_train_data
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def main(CFG):

    modelPath = "distiluse-base-multilingual-cased-v1"
    model_kwargs = {'device':'cuda'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=modelPath,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

    # Retriever
    retriever = db.as_retriever(search_kwargs={"k": 4})

    # Load LORA MODEL
    print('load : ', CFG.new_model)
    config = PeftConfig.from_pretrained('./checkpoints/'+CFG.new_model)
    base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(base_model, './checkpoints/' + CFG.new_model)
    model.to(CFG.device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" 

    print("=" * 80)
    print("model is in device : " + str(model.device))
    print("=" * 80)

    # RAG pipeline
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, device=CFG.device, torch_dtype=torch.float16)
    hf = HuggingFacePipeline(pipeline=pipe)

    # prompt
    template = """마지막에 질문에 답하려면 다음과 같은 맥락을 사용합니다.

    {context}

    질문: {question}

    답변: """
 
    '''
    conversation = [ {'role': 'user', 'content': template} ] 
    prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    '''
    custom_rag_prompt = PromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | hf
        | StrOutputParser()
    )
    


    # Chunk generation 
    
    for chunk in rag_chain.stream("도배지에 녹은 자국이 발생하는 주된 원인과 그 해결 방법은 무엇인가요?"):
        print(chunk, end="", flush=True)
    
    print("=" * 80)
    print("답변 생성 완료!")
    print("=" * 80)
    
    
    # test data inference
    test = pd.read_csv('./data/test_cleaned.csv')

    preds = []
    for test_question in tqdm(test['Question']):
        # 각 질문 row 별로 대답 저장
        preds_temp = []
      
        # 입력 텍스트를 토큰화하고 모델 입력 형태로 변환
        print("="*80)
        for chunk in rag_chain.stream(test_question):
            preds_temp.append(chunk)
            print(chunk, end="", flush=True)
        print("="*80)
            
        '''    
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
        '''
            
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
    parser.add_argument('--ft', type=bool, default=None)
    parser.add_argument('--rag', type=bool, default=None)
    args = parser.parse_args()

    
    CFG = Config()
    CFG = CFG.from_yaml('./configs/config.yaml')
    CFG_custom = Config()
    CFG_custom = CFG.from_yaml('./configs/' + args.config)
    CFG.update(CFG_custom)
    CFG.device = 'cuda:' + str(args.gpu)
    
    CFG.ft = args.ft
    CFG.rag = args.rag
    print(CFG.ft)
    
    main(CFG)