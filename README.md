# CoreML Conversion of the mxbai-embed-large-v1 sentence embedding model

After extensive testing (and a lot of debugging with ChatGPT), I was able to convert the mxbai-embed-large-v1 model to CoreML and run it mostly on the GPU. The ANE version ran 4x slower for 1000x embeddings than ollama, and was reverted.

I verified the output with ollama:

```
curl http://localhost:11434/api/embeddings -d '{
    "model": "mxbai-embed-large",
        "prompt": "This is a test sentence for the CoreML model"
    }'
```

Environment: Python 3.11
coremltools 8.0
sentence-transformers 3.1.0
transformers 4.44.2

The model itself is available on Huggingface: https://huggingface.co/nbpe97/mxbai-embed-large-v1-CoreML/tree/main