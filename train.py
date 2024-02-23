from modules.utils import Config
import argparse
import wandb
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer


def main(CFG):
    # Load tokenizer and model with QLoRA configuration
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

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        CFG.model_name,
        quantization_config=bnb_config,
        device_map=CFG.device_map
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

    # Load LoRA configuration
    peft_config = LoraConfig(
            r = CFG.lora.lora_r,
            lora_alpha = CFG.lora.lora_alpha,
            lora_dropout = CFG.lora.lora_dropout,
            target_modules = CFG.lora.target_modules,
            task_type = CFG.lora.task_type,
            bias = 'none'
        )

    # Set training parameters
    training_arguments = TrainingArguments(
        output_dir=CFG.Train_args.output_dir,
        num_train_epochs=CFG.Train_args.num_train_epochs,
        per_device_train_batch_size=CFG.Train_args.per_device_train_batch_size,
        gradient_accumulation_steps=CFG.Train_args.gradient_accumulation_steps,
        optim=CFG.Train_args.optim,
        save_strategy=CFG.Train_args.save_strategy,
        save_steps=CFG.Train_args.save_steps,
        logging_steps=CFG.Train_args.logging_steps,
        learning_rate=CFG.Train_args.learning_rate,
        weight_decay=CFG.Train_args.weight_decay,
        fp16=CFG.Train_args.fp16,
        bf16=CFG.Train_args.bf16,
        max_grad_norm=CFG.Train_args.max_grad_norm,
        max_steps=CFG.Train_args.max_steps,
        warmup_ratio=CFG.Train_args.warmup_ratio,
        group_by_length=CFG.Train_args.group_by_length,
        lr_scheduler_type=CFG.Train_args.lr_scheduler_type,
        do_eval=True,                    
        evaluation_strategy=CFG.Train_args.save_strategy,
        report_to="wandb"
    )

    print("=" * 80)
    print("Training arg SET")
    print("=" * 80)
    data_files = {"train": f"train_data_{CFG.dataset_name}.csv", "validation": f"valid_data_{CFG.dataset_name}.csv"}
    dataset = load_dataset("./data", data_files=data_files)
    print("=" * 80)
    print("Load Dataset")
    print(dataset['train'][CFG.SFT.dataset_text_field][0])
    print("=" * 80)
    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        peft_config=peft_config,
        dataset_text_field=CFG.SFT.dataset_text_field,
        max_seq_length=CFG.SFT.max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=CFG.SFT.packing,
    )

    print("=" * 80)
    print("START TRAINING")
    print("=" * 80)

    trainer.train()

    trainer.model.save_pretrained(CFG.new_model)
    print("=" * 80)
    print("MODEL SAVED : " + CFG.new_model)
    print("=" * 80)

    # Ignore warnings
    logging.set_verbosity(logging.CRITICAL)

if __name__ == '__main__':
    # python train.py --config gemma.yaml --gpu 2
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="config.yaml")
    parser.add_argument('--gpu', type=str, default= '0,1,2,3')

    args = parser.parse_args()

    CFG = Config()
    CFG = CFG.from_yaml('./configs/config.yaml')
    CFG_custom = Config()
    CFG_custom = CFG.from_yaml('./configs/' + args.config)
    CFG.update(CFG_custom)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print("=" * 80)
    print("Device COUNT : ", torch.cuda.device_count())
    print("Device Num : ", args.gpu)

    print("=" * 80)
    print(CFG)
    
    wandb.login()
    wandb.init(project=CFG.wandb.project_name)
    wandb.run.name = CFG.new_model
    wandb.config.update(CFG)
    wandb.run.save()
    
    main(CFG)