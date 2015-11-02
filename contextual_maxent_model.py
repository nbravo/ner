import numpy as np
import scipy
import sys
from sklearn import linear_model
from collections import defaultdict
import preprocess
from sklearn.externals import joblib

tags = {"UNKNOWN": 0, "O": 1, "B": 2, "I": 3, "START": 4}
reverse_tags = {0: "UNKNOWN", 1: "O", 2: "B", 3: "I"}
word_dict = None
numFeatures = None

if __name__ == "__main__":
  train = sys.argv[1]
  test = sys.argv[2]
  model_number = sys.argv[3]
  X = None
  y = None

  retagged_train, word_dict, totalNumWords = preprocess.iob(train)
  global numFeatures
  numDistinctWords = len(word_dict)
  numFeatures = numDistinctWords + 1 + 9 + 12 + 2
  X = scipy.sparse.lil_matrix((totalNumWords, numFeatures))
  y = np.empty([totalNumWords, ])
  #training_seen = [0] * totalNumWords
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
    phi = None
    rowIndex = 0
    yIndex = 0
    onePrevCounts = [0, 0, 0]
    onePrevPhi = [0, 0, 0, 0]
    onePrevIndex = -1
    onePrevTag = "START"
    twoPrevCounts = [0, 0, 0]
    twoPrevPhi = [0, 0, 0, 0]
    twoPrevIndex = -1
    twoPrevTag = "START"
    for line in f:
      if not identifier_line:
        split_line = line.rstrip('\n').split(" ")
        for token in split_line:
          word, tag = token.split("_")

          counts, index = word_dict[word.lower()]
          length = len(word)
          alpha = 1 if word.isalpha() else 0
          allCaps = 1 if word.upper() == word else 0
          hasNumeric = 1 if any(c.isdigit() for c in word) else 0

          oneHot = [0] * (len(word_dict) + 1)
          oneHot[index] = 1
          oneHot[onePrevIndex] += 1
          oneHot[twoPrevIndex] += 1
          phi = [length, alpha, allCaps, hasNumeric]
          X[rowIndex] = np.array(oneHot + counts + phi + onePrevCounts + onePrevPhi + [tags[onePrevTag]] + twoPrevCounts + twoPrevPhi + [tags[twoPrevTag]])
          rowIndex += 1
          #training_seen[index] = 1

          y[yIndex] = np.array(tags[tag])
          yIndex += 1

          twoPrevCounts = onePrevCounts
          twoPrevPhi = onePrevPhi
          twoPrevIndex = onePrevIndex
          twoPrevTag = onePrevTag

          onePrevCounts = counts
          onePrevPhi = phi
          onePrevIndex = index
          onePrevTag = tag
      identifier_line = not identifier_line
      onePrevCounts = [0, 0, 0]
      onePrevPhi = [0, 0, 0, 0]
      onePrevIndex = -1
      onePrevTag = "START"
      twoPrevCounts = [0, 0, 0]
      twoPrevPhi = [0, 0, 0, 0]
      twoPrevIndex = -1
      twoPrevTag = "START"

  X = X.asformat("csr")

  model = linear_model.LogisticRegression()
  model.fit(X, y)

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
  onePrevCounts = [0, 0, 0]
  onePrevPhi = [0, 0, 0, 0]
  onePrevIndex = -1
  onePrevTag = "START"
  twoPrevCounts = [0, 0, 0]
  twoPrevPhi = [0, 0, 0, 0]
  twoPrevIndex = -1
  twoPrevTag = "START"
  testName, testExt = test.split(".")
  testRetagged = testName + "_retagged." + testExt
  identifier_line = True
  new_line = None
  word = None
  tag = None
  prediction = None
  phi = None
  with open(test) as f, open(testRetagged, 'w') as g:
    for line in f:
      if identifier_line:
        g.write(line)
      else:
        split_line = line.split(" ")
        new_line = []
        prev_tag = None
        for token in split_line:
          word, tag = token.rstrip('\n').split("_")


          length = len(word)
          alpha = 1 if word.isalpha() else 0
          allCaps = 1 if word.upper() == word else 0
          hasNumeric = 1 if any(c.isdigit() for c in word) else 0


          counts = [0, 0, 0]
          index = -10
          oneHot = [0] * (len(word_dict) + 1)
          if word.lower() in word_dict:
            counts, index = word_dict[word.lower()]
          if index >= -1:
            oneHot[index] = 1
          oneHot[onePrevIndex] += 1
          oneHot[twoPrevIndex] += 1
          phi = [length, alpha, allCaps, hasNumeric]

          x = np.array(oneHot + counts + phi + onePrevCounts + onePrevPhi + [tags[onePrevTag]] + twoPrevCounts + twoPrevPhi + [tags[twoPrevTag]])

          prediction = model.predict(x)
          predictedTag = reverse_tags[prediction[0]] # Should be a string
          new_line.append(word + "_" + predictedTag)

          twoPrevCounts = onePrevCounts
          twoPrevPhi = onePrevPhi
          twoPrevIndex = onePrevIndex
          twoPrevTag = onePrevTag

          onePrevCounts = counts
          onePrevPhi = phi
          onePrevIndex = index
          onePrevTag = predictedTag
        g.write(' '.join(new_line) + '\n')
      identifier_line = not identifier_line
      onePrevCounts = [0, 0, 0]
      onePrevPhi = [0, 0, 0, 0]
      onePrevIndex = -1
      onePrevTag = "START"
      twoPrevCounts = [0, 0, 0]
      twoPrevPhi = [0, 0, 0, 0]
      twoPrevIndex = -1
      twoPrevTag = "START"

  split_line = None
  word = None
  tag = None
  new_line = []
  newTag = None
  prev_tag = None
  # At this point, all tags will be IOB. Want to convert to GENE1/2, TAG
  with open(testRetagged) as h:
    identifier_line = True
    for line in h:
      if identifier_line:
        print line.rstrip('\n')
      else:
        split_line = line.split(" ")
        new_line = []
        prev_tag = None
        for token in split_line:
          word, tag = token.rstrip('\n').split("_")
          if tag == "O":
            newTag = "TAG"
          if tag == "I":
            if prev_tag == "GENE1" or prev_tag == "GENE2":
              newTag = prev_tag
            else:
              newTag = "GENE1"
              prev_tag = "GENE1"
          if tag == "B":
            if prev_tag == "GENE1":
              newTag = "GENE2"
              prev_tag = "GENE2"
            else:
              newTag = "GENE1"
              prev_tag = "GENE1"
          new_line.append(word + "_" + newTag)
        print ' '.join(new_line)
      identifier_line = not identifier_line

