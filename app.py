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

text = """
Keyphrase extraction is a technique in text analysis where you extract the
important keyphrases from a document. Thanks to these keyphrases humans can
understand the content of a text very quickly and easily without reading it
completely. Keyphrase extraction was first done primarily by human annotators,
who read the text in detail and then wrote down the most important keyphrases.
The disadvantage is that if you work with a lot of documents, this process
can take a lot of time. 

Here is where Artificial Intelligence comes in. Currently, classical machine
learning methods, that use statistical and linguistic features, are widely used
for the extraction process. Now with deep learning, it is possible to capture
the semantic meaning of a text even better than these classical methods.
Classical methods look at the frequency, occurrence and order of words
in the text, whereas these neural approaches can capture long-term
semantic dependencies and context of words in a text.
""".replace("\n", " ")

keyphrases = extractor(text)

print(keyphrases)


def keyphrases_out(input):
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

    page = wiki.page("")
    return page.summary




    # for k in keyphrases:
    #     page = wiki.page(k)
    #     if page.exists():
    #         break
    # return page.summary

# =====[ DEFINE INTERFACE ]===== #'
# demo = gr.Interface(fn=wikipedia_search, inputs = "text", outputs = "text")
# demo.launch(share=True)








