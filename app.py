import os
import gradio as gr
import numpy as np
import wikipediaapi as wk
import wikipedia
from transformers import (
    TokenClassificationPipeline,
    AutoModelForTokenClassification,
    AutoTokenizer,
    BertForQuestionAnswering,
    BertTokenizer
)
from transformers.pipelines import AggregationStrategy
import torch

# =====[ DEFINE PIPELINE ]===== #
class KeyphraseExtractionPipeline(TokenClassificationPipeline):
    def __init__(self, model, *args, **kwargs):
        super().__init__(
            model=AutoModelForTokenClassification.from_pretrained(model),
            tokenizer=AutoTokenizer.from_pretrained(model),
            *args,
            **kwargs
        )

    def postprocess(self, model_outputs):
        results = super().postprocess(
            model_outputs=model_outputs,
            aggregation_strategy=AggregationStrategy.SIMPLE,
        )
        return np.unique([result.get("word").strip() for result in results])

# =====[ LOAD PIPELINE ]===== #
keyPhraseExtractionModel = "ml6team/keyphrase-extraction-kbir-inspec"
extractor = KeyphraseExtractionPipeline(model=keyPhraseExtractionModel)
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

def keyphrases_extraction(text: str) -> str:
    keyphrases = extractor(text)
    return keyphrases

def wikipedia_search(input: str) -> str:
    input = input.replace("\n", " ")
    keyphrases = keyphrases_extraction(input)

    wiki = wk.Wikipedia('en')
    
    try :
        if len(keyphrases) == 0:
            return "Can you add more details to your question?"
    
        query_suggestion = wikipedia.suggest(keyphrases[0])
        if(query_suggestion != None):
            results = wikipedia.search(query_suggestion)
        else:
            results = wikipedia.search(keyphrases[0])

        index = 0
        page = wiki.page(results[index])
        while not ('.' in page.summary) or not page.exists():
            index += 1
            if index == len(results):
                raise Exception
            page = wiki.page(results[index])
        return page.summary
    
    except:
        return "I cannot answer this question"
    
def answer_question(question):

    context = wikipedia_search(question)
    if (context == "I cannot answer this question") or (context == "Can you add more details to your question?"):
        return context

    # ======== Tokenize ========
    # Apply the tokenizer to the input text, treating them as a text-pair.

    input_ids = tokenizer.encode(question, context)
    question_ids = input_ids[:input_ids.index(tokenizer.sep_token_id)+1]

    # Report how long the input sequence is. if longer than 512 tokens divide it multiple sequences
    length_of_group = 512 - len(question_ids)
    input_ids_without_question = input_ids[input_ids.index(tokenizer.sep_token_id)+1:]
    print(f"Query has {len(input_ids)} tokens, divided in {len(input_ids_without_question)//length_of_group + 1}.\n")

    input_ids_split = []
    for group in range(len(input_ids_without_question)//length_of_group + 1):
        input_ids_split.append(question_ids + input_ids_without_question[length_of_group*group:length_of_group*(group+1)-1])
    input_ids_split.append(question_ids + input_ids_without_question[length_of_group*(len(input_ids_without_question)//length_of_group + 1):len(input_ids_without_question)-1])
    
    scores = []
    for input in input_ids_split:
    # ======== Set Segment IDs ========
    # Search the input_ids for the first instance of the `[SEP]` token.
        sep_index = input.index(tokenizer.sep_token_id)

    # The number of segment A tokens includes the [SEP] token istelf.
        num_seg_a = sep_index + 1

    # The remainder are segment B.
        num_seg_b = len(input) - num_seg_a

    # Construct the list of 0s and 1s.
        segment_ids = [0]*num_seg_a + [1]*num_seg_b

    # There should be a segment_id for every input token.
        assert len(segment_ids) == len(input)

    # ======== Evaluate ========
    # Run our example through the model.
        outputs = model(torch.tensor([input]), # The tokens representing our input text.
                    token_type_ids=torch.tensor([segment_ids]), # The segment IDs to differentiate question from answer_text
                    return_dict=True) 

        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

        max_start_score = torch.max(start_scores)
        max_end_score = torch.max(end_scores)

        print(max_start_score)
        print(max_end_score)

    # ======== Reconstruct Answer ========
    # Find the tokens with the highest `start` and `end` scores.
    
        answer_start = torch.argmax(start_scores)
        answer_end = torch.argmax(end_scores)

    
    # Get the string versions of the input tokens.
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Start with the first token.
        answer = tokens[answer_start]

    # Select the remaining answer tokens and join them with whitespace.
        for i in range(answer_start + 1, answer_end + 1):
        
        # If it's a subword token, then recombine it with the previous token.
            if tokens[i][0:2] == '##':
                answer += tokens[i][2:]
        
        # Otherwise, add a space then the token.
            else:
                answer += ' ' + tokens[i]

        scores.append((max_start_score, max_end_score, answer))

    # Compare scores for answers found and each paragraph and pick the most relevant.

    final_answer = max(scores, key=lambda x: x[0] + x[1])[2]

    return final_answer

# =====[ DEFINE INTERFACE ]===== #'
title = "Azza Knowledge Agent"
examples = [
    ["Where is the Eiffel Tower?"],
    ["What is the population of France?"]
]
demo = gr.Interface(
    title = title,

    fn=answer_question,
    inputs = "text", 
    outputs = "text",
    examples=examples,
    allow_flagging="never",
    )

if __name__ == "__main__":
    demo.launch()