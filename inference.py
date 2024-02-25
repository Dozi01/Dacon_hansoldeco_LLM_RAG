from modules.utils import Config
import argparse

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

from peft import PeftConfig, PeftModel
from tqdm import tqdm

import pandas as pd
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
from modules.postprocess import concat_answer

def format_docs(docs):
    formatted_docs = ''
    for doc in docs:
        print(doc.page_content)
        answer_start = doc.page_content.find('답변: ') + len('답변: ')
        print(answer_start)
        answer_only = doc.page_content[answer_start:]
        print(answer_only)
        formatted_docs += answer_only + '\n'
    return formatted_docs

def main(CFG):

    # Load LORA MODEL
    print('load : ', CFG.new_model)
    config = PeftConfig.from_pretrained('./checkpoints/'+CFG.new_model)
    base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)

    if CFG.ft == True:
        model = PeftModel.from_pretrained(base_model, './checkpoints/' + CFG.new_model)
    else:
        model = base_model

    model.to(CFG.device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" 

    print("=" * 80)
    print("model is in device : " + str(model.device))
    print("=" * 80)

    retriver_modelPath = "distiluse-base-multilingual-cased-v1"
    retriver_model_kwargs = {'device':CFG.device}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=retriver_modelPath,
        model_kwargs=retriver_model_kwargs,
        encode_kwargs=encode_kwargs
    )

    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

    # Retriever
    retriever = db.as_retriever(search_kwargs={"k": 4})

    # RAG pipeline
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, device=CFG.device, torch_dtype=torch.float16)
    hf = HuggingFacePipeline(pipeline=pipe)

    # prompt
    template = """마지막에 질문에 답하려면 다음과 같은 맥락을 사용합니다.
    {context}
    <start_of_turn>user
    질문 : {question}
    답변 : <end_of_turn><start_of_turn>model"""

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

    
    # test data inference
    test = pd.read_csv('./data/test_cleaned.csv')

    generated_answers = []
    for i in tqdm(range(len(test))):
        # 각 질문 row 별로 대답 저장
        test_question = test.at[i,'Question']
        test_id = test.at[i,'id']
        gen_answer = []
      
        # generation for rag
        docs = retriever.get_relevant_documents(test_question)
        formatted_docs = format_docs(docs)
        prompt = f'''아래 정보를 사용하여 질문에 답하시오.
        {formatted_docs}
        <start_of_turn>user\n질문 : {test_question} 답변 : <end_of_turn><start_of_turn>model'''

        input_ids = tokenizer.encode(prompt, return_tensors='pt')


        # for chunk in rag_chain.stream(test_question):
        #     gen_answer.append(chunk)

        # generation without rag
        # prompt = '<start_of_turn>user\n질문 : ' + test_question + ' 답변 : <end_of_turn><start_of_turn>model'
        # input_ids = tokenizer.encode(prompt, return_tensors='pt')

        # 답변 생성
        output_sequences = model.generate(
            input_ids=input_ids.to(CFG.device),
            max_length=4096,
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
            # answer_start = full_text.find(tokenizer.eos_token) + len(tokenizer.eos_token)
            # answer_only = full_text[answer_start:].strip()
            # answer_only = answer_only.replace('\n', ' ')
            gen_answer.append(full_text)

        print("="*80)
        print(prompt)
        print('='* 80)
        print(gen_answer)
        # you can add retrivered document as
        # row = {'id' : test_id, 'answer' : gen_answer, 'retrived document' : documet}
        row = {'id' : test_id, 'answer' : gen_answer}
        generated_answers.append(row)

    answer_df = pd.DataFrame(generated_answers)
    file_name = f'{CFG.new_model}_rag_{CFG.rag}_ft_{CFG.ft}.csv'

    answer_df.to_csv(f'./submission/{file_name}', index = None)
    print("=" * 80)
    print(f"Answer file Saved : {file_name}")

    # concat answers groupby 'id'
    submission_df = concat_answer(answer_df)
    submission_df.to_csv(f'./submission/{file_name}_embeddings', index = None)
    print(f"Submission file Saved : {file_name}_embeddings")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--ft', type=bool, default=True)
    parser.add_argument('--rag', type=bool, default=True)
    args = parser.parse_args()

    
    CFG = Config()
    CFG = CFG.from_yaml('./configs/config.yaml')
    CFG_custom = Config()
    CFG_custom = CFG.from_yaml('./configs/' + args.config)
    CFG.update(CFG_custom)
    CFG.device = 'cuda:' + str(args.gpu)
    
    CFG.ft = args.ft
    CFG.rag = args.rag
    
    main(CFG)