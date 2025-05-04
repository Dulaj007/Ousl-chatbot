import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.optim import AdamW

# Prepare the dataset
train_data = [
    ("Hi there", "greeting"),
    ("Hello!", "greeting"),
    ("Good morning", "greeting"),
    ("Goodbye!", "goodbye"),
    ("See you later", "goodbye"),
    ("Thanks a lot", "thanks"),
    ("Thank you so much", "thanks"),
]

intent_labels = {"greeting": 0, "goodbye": 1, "thanks": 2}
reverse_labels = {v: k for k, v in intent_labels.items()}

# Tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

# Define the Dataset class
class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Prepare the data
texts = [item[0] for item in train_data]
labels = [intent_labels[item[1]] for item in train_data]

train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)

train_dataset = IntentDataset(train_texts, train_labels, tokenizer)
val_dataset = IntentDataset(val_texts, val_labels, tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # number of training epochs
    per_device_train_batch_size=8,   # batch size for training
    per_device_eval_batch_size=8,    # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

# Initialize Trainer
trainer = Trainer(
    model=model,                         # the model to be trained
    args=training_args,                  # training arguments
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset,            # evaluation dataset
    data_collator=None,                  # collator for batching
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

# Example: Test the fine-tuned model
def predict_intent(text):
    input_ids = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(input_ids)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    predicted_label = probs.argmax().item()
    
    return reverse_labels.get(predicted_label, "unknown")

# Testing the model
print(predict_intent("Hi there"))  # Expected: greeting
print(predict_intent("Goodbye!"))  # Expected: goodbye
print(predict_intent("Thank you")) # Expected: thanks
