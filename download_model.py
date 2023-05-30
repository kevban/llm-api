from transformers import pipeline, AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoModelForCausalLM, AutoModelForQuestionAnswering


def download():
    models_to_download = [
        ("cross-encoder/nli-roberta-base", "class"),
        ("distilbert-base-uncased-finetuned-sst-2-english", "sent"),
        ("distilbert-base-uncased-distilled-squad", "qna"),
        ("aisquared/dlite-v2-1_5b", "complete"),
    ]

    for (model_name, model_type) in models_to_download:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model_type == "sent" or model_type == "class":
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            model.save_pretrained("models/" + model_name)
        elif model_type == "complete":
            model = AutoModelForCausalLM.from_pretrained(model_name)
            model.save_pretrained("models/" + model_name)
        elif model_type == "qna":
            model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            model.save_pretrained("models/" + model_name)
        tokenizer.save_pretrained("models/" + model_name)
        print("model " + model_name + "saved")
    
