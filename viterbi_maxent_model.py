import numpy as np
import scipy
import sys
from sklearn import linear_model
from collections import defaultdict
import preprocess
from sklearn.externals import joblib

tags = {"UNKNOWN": 0, "O": 1, "B": 2, "I": 3, "START": 4}
reverse_tags = {0: "UNKNOWN", 1: "O", 2: "B", 3: "I"}
all_classes = np.array([1, 2, 3])
word_dict = None
numFeatures = None

def predict_proba_ordered(probs, classes_, all_classes):
    """
    probs: list of probabilities, output of predict_proba
    classes_: clf.classes_
    all_classes: all possible classes (superset of classes_)
    """
    proba_ordered = np.zeros((probs.shape[0], all_classes.size),  dtype=np.float)
    sorter = np.argsort(all_classes) # http://stackoverflow.com/a/32191125/395857
    idx = sorter[np.searchsorted(all_classes, classes_, sorter=sorter)]
    proba_ordered[:, idx] = probs
    return proba_ordered

def backtrack(dictO, dictB, dictI, lastTag):
  currIndex = len(dictO) - 1
  currTag = lastTag
  tags = []
  score = None
  nextTag = None
  while currIndex > 0:
    if currTag == "O":
      score, nextTag,c,p,w = dictO[currIndex]
    if currTag == "B":
      score, nextTag,c,p,w = dictB[currIndex]
    if currTag == "I":
      score, nextTag,c,p,w = dictI[currIndex]
    tags.append(nextTag)
    currTag = nextTag
    currIndex -= 1
  return tags

def viterbi_decode(line, model):
  split_line = line.split(" ")

  deltaO = {}
  deltaB = {}
  deltaI = {}

  print "Starting first viterbi. Length of this line is ", len(split_line)
  scoreO, t, c, p, w = viterbi(split_line, len(split_line)-1, "O", deltaO, deltaB, deltaI)
  print "finished first viterbi"
  scoreB, t, c, p, w = viterbi(split_line, len(split_line)-1, "B", deltaO, deltaB, deltaI)
  print "finished second viterbi"
  scoreI, t, c, p, w = viterbi(split_line, len(split_line)-1, "I", deltaO, deltaB, deltaI)
  print "finished all viterbi"

  lastTag = None
  if scoreO >= scoreB and scoreO >= scoreI:
    lastTag = "O"
  elif scoreB >= scoreI and scoreB >= scoreO:
    lastTag = "B"
  else:
    lastTag = "I"

  print 'starting backtrack'
  reversedTagSequence = [lastTag] + backtrack(deltaO, deltaB, deltaI, lastTag)
  print 'finished backtrakc'
  correctTagSequence = reversedTagSequence[::-1]

  new_line = []
  for i in xrange(len(split_line)):
    word, tag = split_line[i].rstrip('\n').split("_")
    new_line.append(word + "_" + correctTagSequence[i])
  return ' '.join(new_line) + '\n'

def viterbi(split_line, index, tag, deltaO, deltaB, deltaI):
  if tag == "O":
    if index in deltaO:
      return deltaO[index]
  if tag == "B":
    if index in deltaB:
      return deltaB[index]
  if tag == "I":
    if index in deltaI:
      return deltaI[index]
  word, tag = split_line[index].split("_")
  counts = [0, 0, 0]
  wordIndex = -2

  oneHot = [0] * (len(word_dict) + 1)
  if word.lower() in word_dict:
    counts, wordIndex = word_dict[word.lower()]
  if wordIndex >= -1:
    oneHot[wordIndex] = 1
  length = len(word)
  alpha = 1 if word.isalpha() else 0
  allCaps = 1 if word.upper() == word else 0
  hasNumeric = 1 if any(c.isdigit() for c in word) else 0

  phi = [length, alpha, allCaps, hasNumeric]

  prevO = None
  prevB = None
  prevI = None

  prevCounts = None
  prevPhi = None
  prevIndex = None
  prevTag = None

  if index == 0:
    prevO = 1
    prevB = 1
    prevI = 1

    prevCounts = [0] * 3
    prevPhi = [0] * 4
    prevIndex = -1
    prevTag = "START"
  else:
    if index-1 in deltaO:
      prevO, prevTag, prevCounts, prevPhi, prevIndex = deltaO[index-1]
    else:
      prevO, prevTag, prevCounts, prevPhi, prevIndex = viterbi(split_line, index-1, "O", deltaO, deltaB, deltaI)
    if index-1 in deltaB:
      prevB, prevTag, prevCounts, prevPhi, prevIndex = deltaB[index-1]
    else:
      prevB, prevTag, prevCounts, prevPhi, prevIndex = viterbi(split_line, index-1, "B", deltaO, deltaB, deltaI)
    if index-1 in deltaI:
      prevI, prevTag, prevCounts, prevPhi, prevIndex = deltaI[index-1]
    else:
      prevI, prevTag, prevCounts, prevPhi, prevIndex = viterbi(split_line, index-1, "I", deltaO, deltaB, deltaI)

  oneHot[prevIndex] += 1
  x = np.array(oneHot + counts + phi + prevCounts + prevPhi + [tags[prevTag]])
  first_prediction = predict_proba_ordered(model.predict_proba(x), model.classes_, all_classes)
  prediction_probs = first_prediction[0]

  scoreO = prediction_probs[0] * prevO
  scoreB = prediction_probs[1] * prevB
  scoreI = prediction_probs[2] * prevI

  if scoreO >= scoreB and scoreO >= scoreI:
    if tag == "O":
      deltaO[index] = scoreO, "O", counts, phi, wordIndex
    if tag == "B":
      deltaB[index] = scoreO, "O", counts, phi, wordIndex
    if tag == "I":
      deltaI[index] = scoreO, "O", counts, phi, wordIndex
    return scoreO, "O", counts, phi, wordIndex
  if scoreB >= scoreO and scoreB >= scoreI:
    if tag == "O":
      deltaO[index] = scoreB, "B", counts, phi, wordIndex
    if tag == "B":
      deltaB[index] = scoreB, "B", counts, phi, wordIndex
    if tag == "I":
      deltaI[index] = scoreB, "B", counts, phi, wordIndex
    return scoreB, "B", counts, phi, wordIndex
  else:
    if tag == "O":
      deltaO[index] = scoreI, "I", counts, phi, wordIndex
    if tag == "B":
      deltaB[index] = scoreI, "I", counts, phi, wordIndex
    if tag == "I":
      deltaI[index] = scoreI, "I", counts, phi, wordIndex
    return scoreI, "I", counts, phi, wordIndex



if __name__ == "__main__":
  train = sys.argv[1]
  test = sys.argv[2]
  model_number = sys.argv[3]
  X = None
  y = None

  retagged_train, word_dict, totalNumWords = preprocess.iob(train)
  global numFeatures
  numDistinctWords = len(word_dict)
  numFeatures = numDistinctWords + 1 + 6 + 8 + 1
  X = scipy.sparse.lil_matrix((totalNumWords, numFeatures))
  y = np.empty([totalNumWords, ])
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
          phi = [length, alpha, allCaps, hasNumeric]
          X[rowIndex] = np.array(oneHot + counts + phi + onePrevCounts + onePrevPhi + [tags[onePrevTag]])
          rowIndex += 1

          y[yIndex] = np.array(tags[tag])
          yIndex += 1

          onePrevCounts = counts
          onePrevPhi = phi
          onePrevIndex = index
          onePrevTag = tag
      identifier_line = not identifier_line
      onePrevCounts = [0, 0, 0]
      onePrevPhi = [0, 0, 0, 0]
      onePrevIndex = -1
      onePrevTag = "START"

  X = X.asformat("csr")

  model = linear_model.LogisticRegression()
  model.fit(X, y)

  testName, testExt = test.split(".")
  testRetagged = testName + "_retagged." + testExt
  identifier_line = True
  with open(test) as f, open(testRetagged, 'w') as g:
    for line in f:
      if identifier_line:
        g.write(line)
      else:
        g.write(viterbi_decode(line, model))
      identifier_line = not identifier_line

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

