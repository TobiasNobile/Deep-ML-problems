def unigram_probability(corpus: str, word: str) -> float:
    tokens = corpus.split(" ")
    return tokens.count(word) / len(tokens)