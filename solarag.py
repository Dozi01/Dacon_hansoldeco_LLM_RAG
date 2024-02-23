from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.chroma import Chroma
from data_loader import load_train_data
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
from tqdm import tqdm
import torch
import pickle
from sentence_transformers import SentenceTransformer

# Load train data
data = load_train_data()

# Load base model
base_model_path = "distiluse-base-multilingual-cased-v1"
base_model_kwargs = {'device':'cuda'}
base_encode_kwargs = {'normalize_embeddings': False}
base_embeddings = HuggingFaceEmbeddings(
    model_name=base_model_path,
    model_kwargs=base_model_kwargs,
    encode_kwargs=base_encode_kwargs
)

# Load RAG model
rag_model_id = "Upstage/SOLAR-10.7B-Instruct-v1.0"
rag_tokenizer = AutoTokenizer.from_pretrained(rag_model_id)
rag_model = AutoModelForCausalLM.from_pretrained(rag_model_id).to('cuda')
rag_pipe = pipeline("text-generation", model=rag_model, tokenizer=rag_tokenizer, max_new_tokens=512, device=2,
   torch_dtype=torch.float16)
rag_hf = HuggingFacePipeline(pipeline=rag_pipe)

# Combine base and RAG model
template = """마지막에 질문에 답하려면 다음과 같은 맥락을 사용합니다.

{context}

질문: {question}

유용한 답변: """
custom_rag_prompt = PromptTemplate.from_template(template)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | hf
    | StrOutputParser()
)

# Load test data
test = pd.read_csv("/home/minahwang2001/data/test.csv")

# Perform inference
preds = []
for test_question in tqdm(test['질문']):
    # Generate answer using Solar model
    output_sequences = solar_model.generate(
        input_ids=test_question.to('cuda'),  # Assuming the question is already tokenized and converted to IDs
        max_length=300,
        temperature=0.9,
        top_k=1,
        top_p=0.9,
        repetition_penalty=1.2,
        do_sample=True,
        num_return_sequences=1
    )

    # Process generated text
    for generated_sequence in output_sequences:
        full_text = solar_tokenizer.decode(generated_sequence, skip_special_tokens=False)
        answer_start = full_text.find(solar_tokenizer.eos_token) + len(solar_tokenizer.eos_token)
        answer_only = full_text[answer_start:].strip()
        answer_only = answer_only.replace('\n', ' ')
        preds.append(answer_only)

# Save results
with open("result.pkl", 'wb') as f:
    pickle.dump(preds, f)


# Encode answers using Sentence Transformers
model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
pred_embeddings = model.encode(preds)

















----------------------------------------------------------------------------
result = []
for i in tqdm(range(len(test))):
    _id = test.at[i,'id']
    _q = test.at[i,'질문']
    _a = []
    for chunk in rag_chain.stream(_q):
        _a.append(chunk)
    result.append({"id":_id, "대답":" ".join(_a)})

# Save results
with open("result.pkl", 'wb') as f:
    pickle.dump(result, f)

# Encode answers using Sentence Transformers
_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
for i in range(len(result)):
    result[i]['embedding'] = _model.encode(result[i]['대답'].replace("\u200b"," "))

# Prepare submission
submission = []
for i in range(len(result)):
    tmp = {"id":result[i]['id'],}
    for j in range(len(result[i]['embedding'])):
        tmp[f'vec_{j}'] = result[i]['embedding'][j]
    submission.append(tmp)

# Save submission
pd.DataFrame(submission).to_csv("/home/minahwang2001/baseline/submission_RAG_SOLAR.csv", index=False)