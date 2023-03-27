---
title: Azza
sdk: gradio
app_file: app.py
models: ["ml6team/keyphrase-extraction-kbir-inspec"]
pinned: false
---

# [Azza](jingxiangmo-azza.hf.space/) - Grounded Q/A Conversational Agent 🤖
> We applied BERT to the Stanford Question Answering Dataset (SQuAD) to achieve *natural and grounded* human-level performance on question answering.

Link to demo: jingxiangmo-azza.hf.space/

<br>

<p align="center">
<img width="734"  alt="Screenshot 2023-03-27 at 10 45 57" src="https://user-images.githubusercontent.com/65676392/227976366-b4560691-28ed-4556-9068-e2d6d4bd8bb9.png">
</p>

<br>

# Pipeline
1. Key-phrase extraction from the user's question [1].
2. Retrieval of the relevant Wikipedia article based on the extracted key-phrase (reference text).
3. Tokenization of the question and context.
4. Evaluation of the model and extraction of answer tokens using BERT [2].
5. Reconstruction of the answer from the tokens.
6. Wrap the answer with a generative language model for a better response.

![Frame 1](https://user-images.githubusercontent.com/65676392/227686139-04784d02-c1b0-4911-a0d4-e878a954e5c2.png)

<br>

# Methodology

BERT (Bidirectional Encoder Representation from Transformers) is a pre-trained language model designed to improve nautral-language understanding in various tasks. BERT can find the answer given a question and reference by learning contextual relationships between words in a a text using bidirectional transformers. 

We fine-tuned BERT for question answering using the Stanford Question Answering Dataset (SQuAD 2.0) [3], which is a reading comprehension dataset for training and evaluation. BERT learns to identify the most relevant information within the reference text to answer a given question.

1. **Tokenization**: The question and reference text from wikipedia are tokenized, which means they are converted into a format that BERT can understand (a sequence of tokens).
2. **Input preparation**: The tokenized question and reference are combined into a single input sequence, with special tokens added to indicate the beginning and end of the question and reference segments.
3. **Contextual understanding**: BERT processes the input sequence using its bidirectional transformer architecture, encoding contextual information about each token (word) in the sequence.
4. **Answer prediction**: BERT generates a probability distribution over the tokens in the reference text for both the start and end positions of the answer. The model identifies the tokens with the highest probabilities for the start and end positions as the answer span.
5. **Answer extraction**: The answer is extracted by combining the tokens from the identified start to end positions.
   
 <br>
 
# References
[1] https://huggingface.co/ml6team/keyphrase-extraction-kbir-inspec <br>
[2] https://arxiv.org/pdf/1810.04805v2.pdf <br>
[3] https://rajpurkar.github.io/SQuAD-explorer/ <br>
