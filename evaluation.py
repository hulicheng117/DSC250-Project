import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
import sacrebleu
from tqdm import tqdm 

def bleu_evaluation(model_path, base_model_name, dataset, max_length=128):
    print("Loading base model...")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name).to("cuda")
    print("Loading fine-tuned model...")
    finetuned_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name).to("cuda")
    finetuned_model = PeftModel.from_pretrained(finetuned_model, model_path).to("cuda")
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    test_data = dataset["test"]
    print("Test dataset loaded. Sample size:", len(test_data))
    def generate_translation(model, text, max_length):
        print(f"Generating translation for: {text[:30]}...") 
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to("cuda")
        with torch.no_grad():
            output = model.generate(**inputs, max_length=max_length)
        return tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    subset_data = test_data[:50]  
    references = [example["zh"] for example in subset_data["translation"]]
    print("Generating predictions for base model...")
    base_predictions = [generate_translation(base_model, example["en"], max_length) for example in tqdm(subset_data["translation"], desc="Base model predictions")]
    print("Generating predictions for fine-tuned model...")
    ft_predictions = [generate_translation(finetuned_model, example["en"], max_length) for example in tqdm(subset_data["translation"], desc="Fine-tuned model predictions")]
    print("Calculating BLEU score...")
    base_bleu = sacrebleu.corpus_bleu(base_predictions, [references])
    ft_bleu = sacrebleu.corpus_bleu(ft_predictions, [references])
    print(f"Base Model BLEU Score: {base_bleu.score:.2f}")
    print(f"Fine-tuned Model BLEU Score: {ft_bleu.score:.2f}")
    return base_bleu.score, ft_bleu.score

if __name__ == "__main__":
    model_path = "./model_lora"  
    dataset = load_dataset("Helsinki-NLP/opus-100", "en-zh")
    print("Starting BLEU evaluation...")
    base_bleu, ft_bleu = bleu_evaluation(model_path, "google-t5/t5-base", dataset)

