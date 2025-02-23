import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
import sacrebleu

def blue_evaluation(model_path, base_model_name, dataset):
    base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
    model = PeftModel.from_pretrained(base_model, model_path).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    test_data = dataset["test"]
    def generate_translation(text):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to("cuda")
        with torch.no_grad():
            output = model.generate(**inputs, max_length=128)
        return tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    references = [example["translation"]["zh"] for example in test_data]
    predictions = [generate_translation(example["translation"]["en"]) for example in test_data]
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    print(f"BLEU Score: {bleu.score:.2f}")
    return bleu.score


if __name__ == "__main__":
    model_path = "./model_lora"  
    dataset=load_dataset("Helsinki-NLP/opus-100", "en-zh")
    bleu_score = blue_evaluation(model_path, "Helsinki-NLP/opus-mt-en-zh", dataset)
