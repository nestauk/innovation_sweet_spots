import nltk.data

tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")


def remove_space_after_comma(text):
    """util function to process search terms with comma"""
    return ",".join([s.strip() for s in text.split(",")])


def check_articles_for_comma_terms(text: str, terms: str):
    terms = [term.strip() for term in terms.split(",")]
    sentences_with_terms = find_sentences_with_terms(text, terms, all_terms=True)
    if len(sentences_with_terms) >= 1:
        return True
    else:
        return False


def find_sentences_with_terms(text, terms, all_terms: bool = True):
    """util function that finds terms in sentences"""
    # split text into sentences
    sentences = tokenizer.tokenize(text)
    # keep sentences with terms
    sentences_with_terms = []
    # number of terms in the query
    n_terms = len(terms)
    for i, sentence in enumerate(sentences):
        terms_detected = 0
        # check all terms
        for term in terms:
            if term in sentence.lower():
                terms_detected += 1
        # check if all terms were found
        if all_terms and (terms_detected == n_terms):
            sentences_with_terms.append(sentence)
        # check if at least one term was found
        elif (all_terms is False) and (terms_deteced > 0):
            sentences_with_terms.append(sentence)
        else:
            pass
    return sentences_with_terms
