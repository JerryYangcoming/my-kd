import torch
from tqdm import tqdm


def compute_perplexity(model, tokenizer, dataset, batch_size=16):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    total_loss = 0.0
    total_tokens = 0

    device = next(model.parameters()).device

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs = tokenizer(batch['text'], return_tensors='pt', truncation=True, padding=True).to(device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            num_tokens = (inputs["input_ids"] != tokenizer.pad_token_id).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    ppl = torch.exp(torch.tensor(total_loss / total_tokens))
    return ppl.item()


from transformers import Trainer, TrainingArguments
from datasets import load_metric


def compute_glue_metrics(model, tokenizer, dataset, task_name, batch_size=32):
    metric = load_metric("glue", task_name)

    def preprocess_function(examples):
        if task_name in ['sst2']:
            return tokenizer(examples['sentence'], truncation=True, padding='max_length', max_length=128)
        else:
            return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding='max_length',
                             max_length=128)

    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(
        [col for col in dataset.column_names if col not in ['input_ids', 'attention_mask', 'label']])
    dataloader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=batch_size)

    all_preds = []
    all_labels = []

    device = next(model.parameters()).device

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'label'}
            labels = batch['label']
            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    results = metric.compute(predictions=all_preds, references=all_labels)
    return results
