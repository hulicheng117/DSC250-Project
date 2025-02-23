import time
import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

def print_trainable_parameters(model):
    trainable_params= 0
    all_param= 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params+=param.numel()
    print(f"Trainable params: {trainable_params} || All params: {all_param} || Trainable%: {100 * trainable_params / all_param}")

def load_data(dataset, tokenizer):
    inputs=[data["en"] for data in dataset["translation"]]
    targets=[data["zh"] for data in dataset["translation"]]
    model_inputs=tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    labels=tokenizer(targets, max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"]=labels["input_ids"]
    return model_inputs

def main():
    device="cuda" if torch.cuda.is_available() else "cpu"
    dataset=load_dataset("Helsinki-NLP/opus-100", "en-zh")
    model_name="Helsinki-NLP/opus-mt-en-zh" 
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    model=AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    lora_config=LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q","k","v"],
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )
    
    model=get_peft_model(model, lora_config).to(device)
    print_trainable_parameters(model)
    
    tokenized_datasets=dataset.map(lambda x: load_data(x, tokenizer), batched=True, remove_columns=["translation"])
    print(f"Training data size: {len(tokenized_datasets['train'])} samples")
    print(f"Validation data size: {len(tokenized_datasets['validation'])} samples")
    print(f"Test data size: {len(tokenized_datasets['test'])} samples" if "test" in tokenized_datasets else "No test set available.")
    
    training_args = TrainingArguments(
    output_dir="./lora_finetuned",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    learning_rate=1e-3,
    num_train_epochs=3,
    fp16=True if torch.cuda.is_available() else False,
    logging_dir="./logs", 
    logging_strategy="steps",  
    logging_steps=50,  
    report_to="tensorboard",  
)
    
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        args=training_args,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
    )
    
    model.config.use_cache=False
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

    
    model.save_pretrained("./model_lora")
    tokenizer.save_pretrained("./model_lora")

if __name__ == "__main__":
    main()