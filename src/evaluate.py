def synonym_substitution(sentence):
    # Placeholder: simple example
    return sentence.replace("child", "kid")

def back_translation(sentence, translate_fn):
    # Simplified placeholder
    return sentence  # replace with real pipeline if needed

def augment(sentence):
    sentence_bt = back_translation(sentence, None)
    sentence_aug = synonym_substitution(sentence_bt)
    return sentence_aug