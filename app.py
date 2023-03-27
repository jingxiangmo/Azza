import os
import gradio as gr
import numpy as np
import wikipediaapi as wk
import wikipedia
import openai
from transformers import (
    TokenClassificationPipeline,
    AutoModelForTokenClassification,
    AutoTokenizer,
    BertForQuestionAnswering,
    BertTokenizer,
)
from transformers.pipelines import AggregationStrategy
import torch
from dotenv import load_dotenv


# =====[ DEFINE PIPELINE ]===== #
class KeyphraseExtractionPipeline(TokenClassificationPipeline):
    def __init__(self, model, *args, **kwargs):
        super().__init__(
            model=AutoModelForTokenClassification.from_pretrained(model),
            tokenizer=AutoTokenizer.from_pretrained(model),
            *args,
            **kwargs,
        )

    def postprocess(self, model_outputs):
        results = super().postprocess(
            model_outputs=model_outputs,
            aggregation_strategy=AggregationStrategy.SIMPLE,
        )
        return np.unique([result.get("word").strip() for result in results])

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# =====[ LOAD PIPELINE ]===== #
keyPhraseExtractionModel = "ml6team/keyphrase-extraction-kbir-inspec"
extractor = KeyphraseExtractionPipeline(model=keyPhraseExtractionModel)
model = BertForQuestionAnswering.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad"
)
tokenizer = BertTokenizer.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad"
)

def wikipedia_search(input: str) -> str:
    """Perform a Wikipedia search using keyphrases.

    Args:
        input (str): The input text.

    Returns:
        str: The summary of the Wikipedia page.
    """

    keyphrases = extractor( input.replace("\n", " "))
    wiki = wk.Wikipedia("en")

    try:
        if len(keyphrases) == 0:
            return "Can you add more details to your question?"

        query_suggestion = wikipedia.suggest(keyphrases[0])
        if query_suggestion is not None:
            results = wikipedia.search(query_suggestion)
        else:
            results = wikipedia.search(keyphrases[0])

        index = 0
        page = wiki.page(results[index])
        while not ("." in page.summary) or not page.exists():
            index += 1
            if index == len(results):
                raise Exception
            page = wiki.page(results[index])
        return page.summary
    except:
        return "I cannot answer this question"


def answer_question(question: str) -> str:
    """Answer the question using the context from the Wikipedia search.

    Args:
        question (str): The input question.

    Returns:
        str: The answer to the question.
    """

    context = wikipedia_search(question)
    if (context == "I cannot answer this question") or (
        context == "Can you add more details to your question?"
    ):
        return context

    # Tokenize and split input
    input_ids = tokenizer.encode(question, context)
    question_ids = input_ids[: input_ids.index(tokenizer.sep_token_id) + 1]

    # Report how long the input sequence is. if longer than 512 tokens divide it multiple sequences
    length_of_group = 512 - len(question_ids)
    input_ids_without_question = input_ids[
        input_ids.index(tokenizer.sep_token_id) + 1 :
    ]

    input_ids_split = []
    for group in range(len(input_ids_without_question) // length_of_group + 1):
        input_ids_split.append(
            question_ids
            + input_ids_without_question[
                length_of_group * group : length_of_group * (group + 1) - 1
            ]
        )
    input_ids_split.append(
        question_ids
        + input_ids_without_question[
            length_of_group
            * (len(input_ids_without_question) // length_of_group + 1) : len(
                input_ids_without_question
            )
            - 1
        ]
    )
    scores = []
    for input in input_ids_split:
        # set Segment IDs
        # Search the input_ids for the first instance of the `[SEP]` token.
        sep_index = input.index(tokenizer.sep_token_id)
        num_seg_a = sep_index + 1
        segment_ids = [0] * num_seg_a + [1] * (len(input) - num_seg_a)
        assert len(segment_ids) == len(input)

        # evaulate the model
        outputs = model(
            torch.tensor([input]),
            token_type_ids=torch.tensor(
                [segment_ids]
            ), 
            return_dict=True,
        )

        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

        max_start_score = torch.max(start_scores)
        max_end_score = torch.max(end_scores)

        print(max_start_score)
        print(max_end_score)

        #  reconstruct answer from the tokens
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        answer = tokens[torch.argmax(start_scores)]

        for i in range(torch.argmax(start_scores) + 1, torch.argmax(end_scores) + 1):
            if tokens[i][0:2] == "##":
                answer += tokens[i][2:]
            else:
                answer += " " + tokens[i]
        scores.append((max_start_score, max_end_score, answer))

    # Compare scores for answers found and each paragraph and pick the most relevant.
    answer = max(scores, key=lambda x: x[0] + x[1])[2]

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="Answer the question " + question + "using this answer: " + answer,
        max_tokens=3000,
    )
    return response.choices[0].text.replace("\n\n", " ")

# =====[ DEFINE INTERFACE ]===== #'
title = "Azza - Grounded Q/A Conversational Agent 🤖"
examples = [["Where is the Eiffel Tower?"], ["What is the population of France?"]]
demo = gr.Interface(
    title=title,
    fn=answer_question,
    inputs="text",
    outputs="text",
    examples=examples,
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
