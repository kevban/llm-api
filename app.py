from flask import Flask, request
import torch as Torch
from helper import *
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, AutoModelForQuestionAnswering
import download_model

app = Flask(__name__)

app.config["TRANSFORMERS_OFFLINE"] = 1 #to only use local models downloaded previously
# app.config["FLASK_ENV"] = "development"

@app.route('/download', methods=["POST"])
def download():
    download_model.download()
    return "success"

@app.route('/sentiment', methods=["POST"])
def sentiment():
    """Given content, returns positive/negative
    params:
    query: content
    model: 
        distilbert-base-uncased-finetuned-sst-2-english
    """
    data = request.get_json()
    path = data.get("model", None) or "distilbert-base-uncased-finetuned-sst-2-english"
    path = "models/" + path
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    classifier = pipeline("sentiment-analysis", model=model,
                          tokenizer=tokenizer, return_all_scores=True)

    return classifier(data["query"])


@app.route('/completion', methods=["POST"])
def completion():
    """Completes a sentence
    params:
    query: sentence or instruction
    length: int = 100
    model: 
        aisquared/dlite-v2-1_5b
    """
    data = request.get_json()
    path = data.get("model", None) or "aisquared/dlite-v2-1_5b"
    path = "models/" + path
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, load_in_8bit = True, device_map="auto")
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer,
                         torch_dtype=Torch.bfloat16, trust_remote_code=True, device_map="auto")
    return generator(data["query"], max_length=data.get("length", None) or 100)

@app.route('/question-and-answer', methods=["POST"])
def qna():
    """Answer a question given context and question
    params:
    query: question
    contex: context
    length: int = 100
    model: 
        distilbert-base-uncased-distilled-squad
    """
    data = request.get_json()
    path = data.get("model", None) or "distilbert-base-uncased-distilled-squad"
    path = "models/" + path
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForQuestionAnswering.from_pretrained(path)
    answer_generator = pipeline("question-answering", model=model, tokenizer=tokenizer)
    return answer_generator(question = data["query"], context = data["context"])

@app.route('/classification', methods=["POST"])
def zero_shot_classifier():
    """Performs zero-shot classification from content and a list of labels
    params:
    query: content
    labels: [labels]
    model: 
        cross-encoder/nli-roberta-base
    """
    data = request.get_json()
    path = data.get("model", None) or "cross-encoder/nli-roberta-base"
    path = "models/" + path
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)
    return classifier(data["query"], candidate_labels=data["labels"])