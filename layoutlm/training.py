from transformers import AutoProcessor
from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D, load_from_disk
from PIL import Image

processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
dataset_dict = load_from_disk("layoutlm/dataset")

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
    