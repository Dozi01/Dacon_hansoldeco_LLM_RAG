import os
from modules.utils import Config
import argparse

from peft import PeftConfig, PeftModel
from tqdm import tqdm

import pandas as pd

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.chroma import Chroma
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

from modules.postprocess import concat_answer
from modules.utils import format_docs


def main(CFG):

    compute_dtype = getattr(torch, CFG.bnb.bnb_4bit_compute_dtype)
  
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=CFG.bnb.use_4bit,
        bnb_4bit_quant_type=CFG.bnb.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=CFG.bnb.use_nested_quant,
    )

    # Check GPU compatibility with bfloat16
    if compute_dtype == torch.float16 and CFG.bnb.use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)
    
    # Load LORA MODEL
    print('load : ', CFG.new_model)
    config = PeftConfig.from_pretrained('./checkpoints/'+CFG.new_model)
    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path, 
        quantization_config=bnb_config,
        device_map=CFG.device_map
        )

    if CFG.ft == True:
        model = PeftModel.from_pretrained(base_model, './checkpoints/' + CFG.new_model)
        model.to(CFG.device)
    elif CFG.ft == False:
        model = base_model


    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    print("=" * 80)
    print("model is in device : " + str(model.device))
    print("=" * 80)


    # Retriever
    retriver_modelPath = "distiluse-base-multilingual-cased-v1"
    retriver_model_kwargs = {'device':CFG.device}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=retriver_modelPath,
        model_kwargs=retriver_model_kwargs,
        encode_kwargs=encode_kwargs
    )
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

    retriever = db.as_retriever(search_kwargs={"k": 5})


    # test data inference
    test = pd.read_csv('./data/test_cleaned.csv')

    generated_answers = []
    docs = ''
    for i in tqdm(range(len(test))):
        # 각 질문 row 별로 대답 저장
        test_question = test.at[i,'Question']
        test_id = test.at[i,'id']
        gen_answer = []
      
        # generation for rag
        if CFG.rag == True:
            docs = retriever.get_relevant_documents(test_question)
            formatted_docs = format_docs(docs, CFG.max_tokens//3)

            prompt = f'''<|im_start|>system\nAct like a wallpapering expert. Use the following documents to answer the questions. you must answer the question in Korean.<|im_end|>\n{formatted_docs}<|im_start|>user\n질문 : {test_question} 답변 : <|im_end|>\n<|im_start|>assistant'''

        elif CFG.rag == False:
            prompt = f'''<|im_start|>system\nAct like a wallpapering expert. you must answer the question in Korean.<|im_end|>\n<|im_start|>user\n질문 : {test_question} 답변 : <|im_end|>\n<|im_start|>assistant'''
        input_ids = tokenizer.encode(prompt, return_tensors='pt')

        # 답변 생성
        output_sequences = model.generate(
            input_ids=input_ids.to(CFG.device),
            max_length=CFG.max_token,
            temperature=0.9,
            top_k=1,
            top_p=0.9,
            repetition_penalty=1.5,
            do_sample=True,
            num_return_sequences=1
        )

        # 생성된 텍스트(답변) 저장
        for generated_sequence in output_sequences:
            full_text = tokenizer.decode(generated_sequence, skip_special_tokens=False)
            # 질문과 답변의 사이를 나타내는 '답변 :'을 찾아, 이후부터 출력 해야 함
            answer_start = full_text.find('assistant') + len('assistant')
            answer_only = full_text[answer_start:]
            gen_answer.append(answer_only)

        print("="*80)
        print(prompt)
        print('='* 80)
        print(gen_answer)
        row = {'id' : test_id, 'answer' : gen_answer, 'prompt' : prompt, 'docs' : docs}
        generated_answers.append(row)

    answer_df = pd.DataFrame(generated_answers)
    file_name = f'{CFG.new_model}_rag_{CFG.rag}_ft_{CFG.ft}'

    answer_df.to_csv(f'./submission/{file_name}.csv', index = None)
    print("=" * 80)
    print(f"Answer file Saved : {file_name}.csv")

    # concat answers groupby 'id'
    submission_df = concat_answer(answer_df)
    submission_df.to_csv(f'./submission/{file_name}_embeddings.csv', index = None)
    print(f"Submission file Saved : {file_name}_embeddings.csv")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--gpu', type=str, default=None)
    parser.add_argument('--max_token', type=int, default=1024)

    parser.add_argument('--ft', action=argparse.BooleanOptionalAction, required=True)
    parser.add_argument('--rag', action=argparse.BooleanOptionalAction, required=True)

    args = parser.parse_args()
   
    CFG = Config()
    CFG = CFG.from_yaml('./configs/config.yaml')
    CFG_custom = Config()
    CFG_custom = CFG.from_yaml('./configs/' + args.config)
    CFG.update(CFG_custom)
    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu
    print("=" * 80)
    print('Count of using GPUs:', torch.cuda.device_count())
    print('Current cuda device:', args.gpu, f'({torch.cuda.current_device()})')
    print("=" * 80)
    CFG.device = 'cuda'
    
    CFG.ft = args.ft
    CFG.rag = args.rag
    CFG.max_token = args.max_token

    main(CFG)