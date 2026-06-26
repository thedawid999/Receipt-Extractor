from transformers import AutoProcessor, LayoutLMv3ForTokenClassification, Trainer, TrainingArguments
from transformers.data.data_collator import default_data_collator
from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D, load_from_disk
from PIL import Image
import numpy as np
import evaluate
import json
import torch

processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
dataset_dict = load_from_disk("layoutlm/dataset")

with open("layoutlm/labels.json", "r") as f:
    labels = json.load(f)["labels"]

id2label = {i: label for i, label in enumerate(labels)}
label2id = {label: i for i, label in enumerate(labels)}
label_list = labels

# processor transforms the dataset into model-format
def prepare_examples(examples):
    # load images from image path
    images = [Image.open(path).convert("RGB") for path in examples['image']]
    
    encoded = processor(
        images, 
        examples['tokens'], 
        boxes=examples['bboxes'], 
        word_labels=examples['ner_tags'], 
        padding="max_length", 
        truncation=True,
        return_tensors="np"
    )

    return encoded

# structure to ensure the model gets the right data types
features = Features({
    'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
    'input_ids': Sequence(feature=Value(dtype='int64')),
    'attention_mask': Sequence(Value(dtype='int64')),
    'bbox': Array2D(dtype="int64", shape=(512, 4)),
    'labels': Sequence(feature=Value(dtype='int64')),
})

# applies prepare_examples() one the whole DatasetDict
processed_dataset = dataset_dict.map(
    prepare_examples,
    batched=True,
    remove_columns=['id', 'tokens', 'bboxes', 'ner_tags', 'image'],
    features=features,
)

# transforms processed_dataset in GPU format (PyTorch-Tensors)   
processed_dataset.set_format("torch")

metric = evaluate.load("seqeval")
# important to get metrics for every single label, instead for the whole model
return_entity_level_metrics = False

def get_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    if return_entity_level_metrics:

        final_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"] = v
            else:
                final_results[key] = value
        return final_results
    else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base", id2label=id2label, label2id=label2id)
train_dataset = processed_dataset["train"]
test_dataset = processed_dataset["test"]

training_args = TrainingArguments(
    output_dir="layoutlm/layoutlmv3-finetuned-sroie",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2, # better for complex data
    num_train_epochs=10,
    learning_rate=1e-5,
    weight_decay=0.01,              # prevents overfitting
    eval_strategy="epoch",          # evaluate after each epoch instead after some steps
    save_strategy="epoch",
    logging_steps=10,               # frequent logging for TensorBoard
    load_best_model_at_end=True,
    metric_for_best_model="f1"      # better metric for NER
)   

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    #tokenizer=processor,
    data_collator=default_data_collator,
    compute_metrics=get_metrics,
)

trainer.train()
trainer.evaluate()

trainer.model.save_pretrained('./layoutlm/layoutlmv3-final-v1')
processor.save_pretrained('./layoutlm/layoutlmv3-final-v1')