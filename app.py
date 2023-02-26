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
       
# Load pipeline
model_name = "ml6team/keyphrase-extraction-kbir-inspec"
extractor = KeyphraseExtractionPipeline(model=model_name)


def keyphrases_out(input): #Not used but might be useful
    input = input.replace("\n", " ")
    keyphrases = extractor(input)
    out = "The Key Phrases in your text are:\n\n"
    for k in keyphrases:
        out += k + "\n"
    return keyphrases

def wikipedia_search(input):
    input = input.replace("\n", " ")
    keyphrases = extractor(input)
    wiki = wk.Wikipedia('en')
    for k in keyphrases:
        page = wiki.page(k)
        if page.exists():
            break
    return page.summary


demo = gr.Interface(fn=wikipedia_search, inputs = "text", outputs = "text")

demo.launch()








