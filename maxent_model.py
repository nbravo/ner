import numpy as np
import scipy
import sys
from sklearn import linear_model
from collections import defaultdict
import preprocess
from sklearn.externals import joblib

tags = {"UNKNOWN": 0, "O": 1, "B": 2, "I": 3}
reverse_tags = {0: "UNKNOWN", 1: "O", 2: "B", 3: "I"}
word_dict = None
numFeatures = None

if __name__ == "__main__":
  train = sys.argv[1]
  test = sys.argv[2]
  model_number = sys.argv[3]
  X = None
  y = None

  retagged_train, word_dict = preprocess.iob(train)
  global numFeatures
  numWords = len(word_dict)
  numFeatures = numWords + 3 + 4
  X = scipy.sparse.lil_matrix((numWords, numFeatures))
  y = np.empty([numWords, ])
  training_seen = [0] * numWords
  with open(retagged_train) as f:
    identifier_line = True
    split_line = None
    word = None
    tag = None
    length = None
    alpha = None
    allCaps = None
    hasNumeric = None
    counts = None
    index = None
    oneHot = None
    rowIndex = 0
    yIndex = 0
    for line in f:
      if not identifier_line:
        split_line = line.rstrip('\n').split(" ")
        for token in split_line:
          word, tag = token.split("_")

          counts, index = word_dict[word.lower()]
          if training_seen[index] == 0:
            length = len(word)
            alpha = 1 if word.isalpha() else 0
            allCaps = 1 if word.upper() == word else 0
            hasNumeric = 1 if any(c.isdigit() for c in word) else 0

            oneHot = [0] * len(word_dict)
            oneHot[index] = 1
            X[rowIndex] = np.array(oneHot + counts + [length, alpha, allCaps, hasNumeric])
            rowIndex += 1
            training_seen[index] = 1

            y[yIndex] = np.array(tags[tag])
            yIndex += 1
      identifier_line = not identifier_line

  X = X.asformat("csr")


  model = linear_model.LogisticRegression()
  model.fit(X, y)
  joblib.dump(model, 'model1.pkl')

  testWords = preprocess.getWordDict(test)
  X = scipy.sparse.lil_matrix((len(testWords), numFeatures))
  length = None
  alpha = None
  allCaps = None
  hasNumeric = None
  counts = None
  index = None
  oneHot = None
  words = None
  x = None
  predictions = None
  new_line = None
  rowIndex = 0
  wordIndex = 0
  for word in testWords:
    testWords[word] = wordIndex
    wordIndex += 1
    length = len(word)
    alpha = 1 if word.isalpha() else 0
    allCaps = 1 if word.upper() == word else 0
    hasNumeric = 1 if any(c.isdigit() for c in word) else 0

    counts = [0, 0, 0]
    index = -1
    oneHot = [0] * len(word_dict)
    if word.lower() in word_dict:
      counts, index = word_dict[word.lower()]
    if index >= 0:
      oneHot[index] = 1
    X[rowIndex] = np.array(oneHot + counts + [length, alpha, allCaps, hasNumeric])
    rowIndex += 1
  X = X.asformat("csr")
  predictions = model.predict(X)

  split_line = None
  word = None
  tag = None
  new_line = []
  newPred = None
  prev_tag = None
  with open(test) as f:
    identifier_line = True
    for line in f:
      if identifier_line:
        print line.rstrip('\n')
      else:
        split_line = line.split(" ")
        new_line = []
        prev_tag = None
        for token in split_line:
          word, tag = token.rstrip('\n').split("_")
          prediction = predictions[testWords[word]]
          if prediction == 0 or prediction == 1:
            newPred = "TAG"
          if prediction == 3:
            if prev_tag == "GENE1" or prev_tag == "GENE2":
              newPred = prev_tag
            else:
              newPred = "GENE1"
              prev_tag = "GENE1"
          if prediction == 2:
            if prev_tag == "GENE1":
              newPred = "GENE2"
              prev_tag = "GENE2"
            else:
              newPred = "GENE1"
              prev_tag = "GENE1"
          new_line.append(word + "_" + newPred)
        print ' '.join(new_line)
      identifier_line = not identifier_line

