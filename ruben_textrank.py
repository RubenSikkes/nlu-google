import nltk
import math
import os
import sys
import numpy
import pandas
import copy
import collections
import nltk
import nltk.tokenize

'''
    ruben_textrank.py
    -----------
    This module implements TextRank, an unsupervised keyword
    significance scoring algorithm. TextRank builds a weighted
    graph representation of a document using words as nodes
    and coocurrence frequencies between pairs of words as edge 
    weights. It then applies PageRank to this graph, and 
    treats the PageRank score of each word as its significance.
    The original research paper proposing this algorithm is
    available here:
    
        https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf

    Main function is page_rank
'''


#nltk.download('averaged_perceptron_tagger')

def powerIteration(transitionWeights, rsp=0.15, epsilon=0.00001, maxIterations=1000):
    # Clerical work:
    transitionWeights = pandas.DataFrame(transitionWeights)
    nodes = __extractNodes(transitionWeights)
    transitionWeights = __makeSquare(transitionWeights, nodes, default=0.0)
    transitionWeights = __ensureRowsPositive(transitionWeights)

    # Setup:
    state = __startState(nodes)
    transitionProbs = __normalizeRows(transitionWeights)
    transitionProbs = __integrateRandomSurfer(nodes, transitionProbs, rsp)
    
    # Power iteration:
    for iteration in range(maxIterations):
        oldState = state.copy()
        state = state.dot(transitionProbs)
        delta = state - oldState
        if __euclideanNorm(delta) < epsilon: 
            break

    return state

def __extractNodes(matrix):
    nodes = set()
    for colKey in matrix:
        nodes.add(colKey)
    for rowKey in matrix.T:
        nodes.add(rowKey)
    return nodes
def __makeSquare(matrix, keys, default=0.0):
    matrix = matrix.copy()
    
    def insertMissingColumns(matrix):
        for key in keys:
            if not key in matrix:
                matrix[key] = pandas.Series(default, index=matrix.index)
        return matrix

    matrix = insertMissingColumns(matrix) # insert missing columns
    matrix = insertMissingColumns(matrix.T).T # insert missing rows

    return matrix.fillna(default)

def __ensureRowsPositive(matrix):
    matrix = matrix.T
    for colKey in matrix:
        if matrix[colKey].sum() == 0.0:
            matrix[colKey] = pandas.Series(numpy.ones(len(matrix[colKey])), index=matrix.index)
    return matrix.T

def __normalizeRows(matrix):
    return matrix.div(matrix.sum(axis=1), axis=0)

def __euclideanNorm(series):
    return math.sqrt(series.dot(series))

# PageRank specific functionality:

def __startState(nodes):
    if len(nodes) == 0: raise ValueError("There must be at least one node.")
    startProb = 1.0 / float(len(nodes))
    return pandas.Series({node : startProb for node in nodes})

def __integrateRandomSurfer(nodes, transitionProbs, rsp):
    alpha = 1.0 / float(len(nodes)) * rsp
    return transitionProbs.copy().multiply(1.0 - rsp) + alpha



def __asciiOnly(string):
    return "".join([char if ord(char) < 128 else "" for char in string])

def __isPunctuation(word):
    return word in [".", "?", "!", ",", "\"", ":", ";", "'", "-"]

def __tagPartsOfSpeech(words):
    return [pair[1] for pair in nltk.pos_tag(words)]

def __tokenizeWords(sentence):
    return nltk.tokenize.word_tokenize(sentence)


def __preprocessDocument(document, relevantPosTags, stopwoorden):
    '''
    This function accepts a string representation 
    of a document as input, and returns a tokenized
    list of words corresponding to that document. Also filters stopwords etc (dutch)
    '''
    
    words = __tokenizeWords(document)
    s = set(stopwoorden) # defined above and nl stop words
    words_f = [x for x in words if x not in s]
    posTags = __tagPartsOfSpeech(words_f)
    
    # Filter out words with irrelevant POS tags
    filteredWords = []
    for index, word in enumerate(words_f):
        word = word.lower()
        tag = posTags[index]
        if not __isPunctuation(word) and tag in relevantPosTags:
            filteredWords.append(word)

    return filteredWords


def page_rank(document, stopwoorden, amount=10, rsp=0.15, windowSize=2):
    words = __preprocessDocument(document, ["NN", "ADJ"], stopwoorden=stopwoorden)
    edgeWeights = collections.defaultdict(lambda: collections.Counter())
    windowSize=windowSize
    for index, word in enumerate(words):
        for otherIndex in range(index - windowSize, index + windowSize + 1):
            if otherIndex >= 0 and otherIndex < len(words) and otherIndex != index:
                otherWord = words[otherIndex]
                edgeWeights[word][otherWord] += 1.0
    wordProbabilities = powerIteration(edgeWeights, rsp=0.15)
    return wordProbabilities.sort_values(ascending=False)[:amount]


