import coremltools as ct
from transformers import AutoTokenizer
import numpy as np
import sys

# Load the CoreML model
model = ct.models.MLModel("SentenceTransformer.mlpackage")

# Print input descriptions
#print("Model Inputs:")
#for input in model.input_description:
#    print(f"Name: {input}, Type: {model.input_description[input]}")

# Print output descriptions
#print("\nModel Outputs:")
#for output in model.output_description:
#    print(f"Name: {output}, Type: {model.output_description[output]}")

# Print the full spec (for more details)
#print("\nFull Model Specification:")
print(model.get_spec())

sys.exit()

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("mixedbread-ai/mxbai-embed-large-v1")

# Prepare some input text
input_text = "This is a test sentence for the CoreML model"
inputs = tokenizer(input_text, return_tensors="np", padding=True, truncation=True, max_length=512)

# Extract input tensors
input_ids = inputs['input_ids'].astype(np.float32)  # CoreML expects float32
attention_mask = inputs['attention_mask'].astype(np.float32)

# Prepare inputs for the CoreML model
coreml_input = {"input_ids": input_ids, "attention_mask": attention_mask}

# Set the compute units to "all" to prefer ANE
config = ct.ComputeUnit.ALL

# Perform inference
for i in range(0, 1000):
	predictions = model.predict(coreml_input)
# Get the hidden states (output of the model)
hidden_states = predictions['hidden_states']

# Extract the embedding for the [CLS] token (the first token in the sequence)
cls_embedding = hidden_states[0, 0, :]  # First token embedding (shape: 1024)

# Set NumPy to print the full array without truncation
np.set_printoptions(threshold=np.inf)

# Print the CLS token embedding, which is a 1024-dimensional vector
print("CLS Token Embedding:", cls_embedding, len(cls_embedding))

# Output results
#print(predictions)

#curl http://localhost:11434/api/embeddings -d '{
#  "model": "mxbai-embed-large",
#  "prompt": "This is a test sentence for the CoreML model"
#}'