import numpy as np
from wordfreq import word_frequency

""" Features used 

* Word Embedding [GloVe 50 dimensional embeddings](http://nlp.stanford.edu/data/glove.6B.zip)
* Length of word
* Syllable count [PyPy](https://pypi.org/project/syllables/)
* Word Frequency [PyPy](https://pypi.org/project/wordfreq/)
* POS tag [Spacy](https://spacy.io/usage/linguistic-features#pos-tagging)

  [Reference](https://www.aclweb.org/anthology/W18-0508.pdf)

"""

# Construct features for single word expressions

def prepare_features_single_word(tokens, sentences, nlp, word_to_ix, model, embedding_index, EMBEDDING_DIM):
  features = []
  for idx, word in enumerate(tokens):
    word = word.lower()
    feature = []

    # Word length
    feature.append(len(word))
    doc = nlp(word)

    # Syllable count and word frequency in the corpus
    # Spacy tokenizes the input sentence
    # In this case we would have only one token, the target word
    for token in doc:
      feature.append(token._.syllables_count)
      feature.append(word_frequency(word, 'en'))

    # Probability of target word `word` in the sentence estimated from by `model`
    if word in word_to_ix:
      # Output scores for each of the word in the sentence
      out = model(sentences[idx])
      pos = -1
      for itr, token in enumerate(sentences[idx].split()):
        if token.lower() == word:
          pos = itr
          break
      id_pos = word_to_ix[word] # word to id mapping
      feature.append(float(out[pos][id_pos]))
    else:
      # `word` not in vocabulary, so cannot predict probability in context
      feature.append(0.0)

    # GloVE embedding for the `word`
    if word in embedding_index:
      feature.extend(embedding_index[word].tolist())
    else:
      # `word` not in the GloVE corpus, take a random embedding
      feature.extend(np.random.random(EMBEDDING_DIM).tolist())
    features.append(feature)

    if (idx + 1) % 500 == 0:
      print('Prepared features for {} single target word sentences'.format(idx + 1))
  return features

# Construct features for multi word expressions

def prepare_features_multi_word(tokens, sentences, nlp, word_to_ix_multi, model_multi, embedding_index, EMBEDDING_DIM):
  features = []
  for idx, word in enumerate(tokens):
    word = word.lower()
    feature = []
    doc = nlp(word)
    word = word.split(' ')
    assert(len(word) == 2)

    # MWE length = sum(length of individual words)
    feature.append(len(word[0]) + len(word[1]))

    syllables = 0
    probability = 1
    embedding = np.zeros(EMBEDDING_DIM)

    # Syllable count and word frequency in the corpus
    # Spacy tokenizes the input sentence
    # In this case we would have two tokens

    for token in doc:
      word_ = token.text
      syllables += token._.syllables_count
      probability *= word_frequency(word_, 'en')

      # GloVE embedding current `word_` of the MWE
      if word_ in embedding_index:
        embedding = embedding + embedding_index[word_]
      else:
        # `word_` not in the GloVE corpus, take a random embedding
        embedding = embedding + np.random.random(EMBEDDING_DIM)

    # Average embedding of the two tokens in the MWE
    embedding = embedding / 2
    feature.append(syllables)
    feature.append(probability)

    # Product of probabilities of constituent words in the MWE
    if word[0] in word_to_ix_multi and word[1] in word_to_ix_multi:
      # Output scores for each of the word in the sentence
      out = model_multi(sentences[idx])
      pos0, pos1 = -1, -1
      for itr, token in enumerate(sentences[idx].split()):
        if token.lower() == word[0]:
          pos0 = itr
          pos1 = itr + 1
          break
      id_pos0 = word_to_ix_multi[word[0]]
      id_pos1 = word_to_ix_multi[word[1]]
      feature.append(float(out[pos0][id_pos0] * out[pos1][id_pos1]))
    else:
      # Either of the constituent words of the MWE not in vocabulary \
      # So cannot predict probability in context
      feature.append(0.0)

    feature.extend(embedding.tolist())
    features.append(feature)

    if (idx + 1) % 500 == 0:
      print('Prepared features for {} multi target word sentences'.format(idx + 1))

  return features
