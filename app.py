import os
import gradio as gr
import wikipediaapi as wk
from transformers import (
    TokenClassificationPipeline,
    AutoModelForTokenClassification,
    AutoTokenizer,
)
from transformers.pipelines import AggregationStrategy
import numpy as np

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
model_name = "ml6team/keyphrase-extraction-kbir-inspec"
extractor = KeyphraseExtractionPipeline(model=model_name)

#TODO: add further preprocessing
def keyphrases_extraction(text: str) -> str:
    keyphrases = extractor(text)
    return keyphrases

def wikipedia_search(input: str) -> str:
    input = input.replace("\n", " ")
    keyphrases = keyphrases_extraction(input)
    wiki = wk.Wikipedia('en')
    
    try :
        #TODO: add better extraction and search
        keyphrase_index = 0
        page = wiki.page(keyphrases[keyphrase_index])

        while not ('.' in page.summary) or not page.exists():
            keyphrase_index += 1
            if keyphrase_index == len(keyphrases):
                raise Exception
            page = wiki.page(keyphrases[keyphrase_index])
        return  page.summary
    except:
        return "I cannot answer this question"

# =====[ DEFINE INTERFACE ]===== #'
title = "Azza Chatbot"
examples = [
    ["Where is the Eiffel Tower?"],
    ["What is the population of France?"]
]


demo = gr.Interface(
    title = title,

    fn=wikipedia_search,
    inputs = "text", 
    outputs = "text",

    examples=examples,
    )

if __name__ == "__main__":
    demo.launch(share=True)
