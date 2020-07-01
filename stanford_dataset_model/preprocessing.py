from sentiment_tokenizer import Tokenizer

tok = Tokenizer(preserve_case=False)


def text_preprocessing(sent):
    train_review = []
    for review in sent:
        s = ''
        words = tok.tokenize(review)
        tokenized_words = []
        for word in words:
            tokenized_words.append(word)
        j = 0
        while (j < len(tokenized_words)):
            k = 0
            if (("n't" in tokenized_words[j]) or ("not" in tokenized_words[j]) or (tokenized_words[j] == '.')):
                s += " " + tokenized_words[j]
                index = j + 1
                i = index
                while (i < len(tokenized_words)):
                    if (tokenized_words[i] != 'but' and tokenized_words[i] != 'and' and tokenized_words[i] != '.' and
                            tokenized_words[i] != ','):
                        s += " NOT_" + tokenized_words[i]
                        i += 1
                        k = i
                    else:
                        s += " " + tokenized_words[i]
                        i += 1
                        k = i
                        break
            else:
                s += " " + tokenized_words[j]
            if (k == 0):
                j += 1
            else:
                j = k
        train_review.append(s)
    return train_review
