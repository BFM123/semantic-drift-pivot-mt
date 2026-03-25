from transformers import MarianMTModel, MarianTokenizer

def load_model(model_name):
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

def translate(texts, tokenizer, model):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs)
    return [tokenizer.decode(t, skip_special_tokens=True) for t in outputs]

if __name__ == "__main__":
    ny_en_model = "Helsinki-NLP/opus-mt-ny-en"
    en_hi_model = "Helsinki-NLP/opus-mt-en-hi"

    ny_tokenizer, ny_model = load_model(ny_en_model)
    en_tokenizer, en_model = load_model(en_hi_model)

    sample = ["Mwana ali ndi njala"]
    english = translate(sample, ny_tokenizer, ny_model)
    hindi = translate(english, en_tokenizer, en_model)

    print("English:", english)
    print("Hindi:", hindi)