import sys

word_dict = {}
uniqueIndex = 0
totalNumWords = 0
''' Returns fileName of the retagged file along with a dictionary.
word: [NumSeenO, NumSeenB, NumSeenI]
'''
def iob(fileName):
  outfileContents = []
  with open(fileName) as f:
    identifier_line = True
    for line in f:
      if identifier_line:
        outfileContents.append(line)
      else:
        outfileContents.append(retag(line))
      identifier_line = not identifier_line
  name, ext = fileName.split(".")
  outfileName = name + "_retagged." + ext
  g = open(outfileName, 'w')
  g.write(''.join(outfileContents))
  g.close()
  return outfileName, word_dict, totalNumWords

def getWordDict(fileName):
  scopedWordDict = {}
  with open(fileName) as f:
    identifier_line = True
    for line in f:
      if not identifier_line:
        split_line = line.rstrip('\n').split(" ")
        for token in split_line:
          word,tag = token.split("_")
          scopedWordDict[word] = 0
      identifier_line = not identifier_line
  return scopedWordDict

'''
Input: a_GENE1 b_GENE1 c_TAG d_GENE2
Output: a_B b_I c_O d_B
'''
def retag(line):
  global uniqueIndex
  global totalNumWords
  new_tokens = []
  prev_tag = None
  split_line = line.rstrip('\n').split(" ")
  for token in split_line:
    totalNumWords += 1

    word,tag = token.split("_")
    counts = None
    currIndex = None
    if word.lower() in word_dict:
      counts, currIndex = word_dict[word.lower()]
    else:
      counts = [0, 0, 0]
      currIndex = uniqueIndex
      uniqueIndex += 1
    if tag == "TAG":
      new_tokens.append(word + "_O")
      counts[0] = counts[0] + 1
    else:
      if tag == prev_tag:
        new_tokens.append(word + "_I")
        counts[2] = counts[2] + 1
      else:
        new_tokens.append(word + "_B")
        counts[1] = counts[1] + 1
    word_dict[word.lower()] = counts, currIndex
    prev_tag = tag
  return ' '.join(new_tokens) + '\n'


if __name__ == "__main__":
  iob(sys.argv[1])
