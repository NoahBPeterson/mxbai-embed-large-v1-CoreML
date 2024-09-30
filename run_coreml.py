import coremltools as ct
from transformers import AutoTokenizer
import numpy as np

# Load the CoreML model
model = ct.models.CompiledMLModel("mxbai-embed-large-v1-test-iOS18.mlmodelc")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("mixedbread-ai/mxbai-embed-large-v1")

# Prepare some input text
input_text = "This is a test sentence for the CoreML model"
inputs = tokenizer(input_text, return_tensors="np", padding=True, truncation=True, max_length=512)

# Extract input tensors
input_ids = inputs['input_ids'].astype(np.int32)
attention_mask = inputs['attention_mask'].astype(np.int32)

# Prepare inputs for the CoreML model
coreml_input = {"input_ids": input_ids, "attention_mask": attention_mask}

# Perform inference
#for i in range(0, 1000):
predictions = model.predict([coreml_input] * 1000)
# Get the hidden states (output of the model)
print(predictions)
hidden_states = predictions[0]['hidden_states']

# Extract the embedding for the [CLS] token (the first token in the sequence)
cls_embedding = hidden_states[0, 0, :]  # First token embedding (shape: 1024)

# Set NumPy to print the full array without truncation
np.set_printoptions(threshold=np.inf)

# Print the CLS token embedding, which is a 1024-dimensional vector
print("CLS Token Embedding:", cls_embedding, len(cls_embedding))
