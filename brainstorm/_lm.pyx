from __future__ import division, print_function

cimport numpy as np

import re

# LM rule copied from Biokit
#         /**
#          * history is one shorter than mMaxN
#          * Example:
#          * bigram: mMaxN = 2, length of history is 1
#          * trigram: mMaxN = 3, length of history is 2
#          * 
#          * Search for n-gram score is as follows:
#          * Example trigram with mMaxN = 3
#          * 
#          * highestBackoff = mMaxN - 1 = 2
#          * 1. try to find bigram backoff
#          *      -> if found look for trigram
#          *              ->if we have the trigram return result
#          *              ->trigram was not found add bigram backoff
#          * 2. if we found the correct m-gram we can return the result,
#          *    otherwise start over with highestBackoff being decremented by one -> now looking for unigram backoff and bigram.
#          * 
#          * If no m-gram for wanted token can be found return max score of NumericValue.
#          */

# an LM file:
# \data\
#         ngram 1=48
#         ngram 2=1174
# 
# \1-grams:
#         -1.563456       </s>
#         -99     <s>     -2.051529
#         -1.864486       AA      -2.40901
#         -1.59342        AE      -2.888558
# 
# etc.
# 
# \end\
#

# Exception for wrong file  format
class LanguageModelFileFormatException(Exception):
    pass

# Exception for wrong contents (e.g. trigram found, but prefix not available)
class LanguageModelStatsException(Exception):
    pass

# Node of the LM tree (TODO add documentation)
# there is a root node with empty fields, using for easy traversing
cdef class LanguageModelTreeNode:
    cdef dict children
    cdef str token
    cdef double score
    cdef double backoff

    # make normal node or root node (if token is None)
    def __init__(self,str token='',double score=0,double backoff=0):
        self.children = {} #TODO
        if token == '':
            self.token = None
            # rest not set to early spot errors
        else:
            self.token = token
            self.score = score
            self.backoff = backoff

    cdef addChild(self,LanguageModelTreeNode childNode):
        assert childNode.token not in self.children.keys()
        self.children[childNode.token] = childNode

#     # Score the sequence, possibly using the next backoff
#     # Returns none if even the next backoff wasn't found, then the caller should try
#     # a shorter history (see above). If this is not the root node, it is assumed that the
#     # current token has already been removed from the sequence
#     # The result is in LOG domain
#     def getScoreOrBackoff(self,sequence):
#         if len(sequence) > 1:
#             # traverse, if the next backoff is not found, return None
#             try:
#                 nextNode = self.children[sequence[0]]
#                 return nextNode.getScoreOrBackoff(sequence[1:])
#             except KeyError:
#                 return None
#         elif len(sequence) == 1:
#             # check for child and return score, if not available, return this score + backoff
#             try:
#                 nextNode = self.children[sequence[0]]
#                 return nextNode.score
#             except KeyError:
#                 return self.score + self.backoff
            
    # Return a node belonging to a certain sequence, or raise LanguageModelStatsException if impossible
    def traverse(self,sequence):
#         if len(sequence) == 1:
#             return self.children[sequence[0]]
#         else:
#             return self.children[sequence[0]].traverse(sequence[1:])
        if len(sequence) == 0:
            return self
        else:
            try:
                return self.children[sequence[0]].traverse(sequence[1:])
            except KeyError:
                raise LanguageModelStatsException('traverse: node not found')

    # return the node belonging to a certain sequence, or to a prefix. Return value is a tuple (node,depth).
    # If we could satisfy the entire request, depth == len(sequence)
    def traversePartly(self,sequence,startDepth = 0):
        if len(sequence) == 0:
            return (self,startDepth)
        else:
            try:
                return self.children[sequence[0]].traversePartly(sequence[1:],startDepth + 1)
            except KeyError:
                # child not found - stop here
                return (self,startDepth)

# N-gram language model, currently working with ARPA n-gram format files
cdef class LanguageModel(object):
    cdef LanguageModelTreeNode lmTree
    cdef int maxN
    # Initialize by reading an LM from a file object or a string and building a tree. 
    def __init__(self,lmSource,maxN = None):
        # possibly open file
        if isinstance(lmSource, basestring):
            lmIter = iter(lmSource.split('\n'))
        else:
            lmIter = lmSource

        self.lmTree = LanguageModelTreeNode()

        headerRead = False

#         lmFid = open(lmFile)
        # parse, looking for the markers \data\, \x-grams:, and \end\
        while True:
            line = lmIter.next()
            print('Read line ---%s---' % line)
#             if len(line) == 0:
#                     raise LanguageModelFileFormatException('Reached end of file without \\end\\ marker')
            if line.isspace():
                continue
            if re.match(r'\\data\\',line):
                if headerRead:
                    raise LanguageModelFileFormatException('Duplicate \\data\\ part')
                headerRead = True

                nGramCounts = self.readHeaderPart(lmIter)
                nGramsRead = { x:False for x in nGramCounts.keys() }
            elif re.match(r'\\[0-9]-grams:',line):
                if not headerRead:
                    raise LanguageModelFileFormatException('NGram part reached, but no header found yet')
                n = int(re.match(r'\\([0-9])-grams:',line).groups()[0])
                if n not in nGramsRead.keys():
                    raise LanguageModelFileFormatException('No count given for %d-grams' % n)
                if nGramsRead[n]:
                    raise LanguageModelFileFormatException('Double %d-gram part' % n)
                self.readNGramPart(lmIter,n,nGramCounts[n])
                nGramsRead[n] = True
            elif re.match(r'\\end\\',line):
                print('Mathcing end')
                if not headerRead or any([ not x for x in nGramsRead.values()]):
                    raise LanguageModelFileFormatException('NGram part reached, but no header found yet')
                break
#         lmFid.close()
        
        if maxN is not None:
            self.maxN = min(self.maxN,maxN)

        
    # read the header of an LM file, returning the number of x-grams as a dictionary
    cdef readHeaderPart(self,lmIter):
        result = {}
        while True:
            line = lmIter.next()
            if re.match(r'^\s*$',line):
                self.maxN = max(result.keys())
                return result
            else:
                match = re.match(r'^ngram ([0-9]+)=([0-9]+)',line)
                result[int(match.groups()[0])] = int(match.groups()[1])

    # read an n-gram definition part
    cdef bint readNGramPart(self,lmIter,int n,int totalCount) except False:
        cdef str template
        cdef int count
        cdef str line
        cdef int i

        cdef list seqData
        cdef LanguageModelTreeNode parentNode
        cdef LanguageModelTreeNode newNode

        template = r'([-.0-9]+)'
        template += '('
        for i in range(n):
            template += r'\s+\S+'
        template += ')'
        template += r'\s*(\s[-.0-9]+)?\s*$' # backoff?

        count = 0
        while True: # note that the counting makes this loop finite
            line = lmIter.next()
            if count == totalCount:
                if  len(line) == 0 or line.isspace():
                    return True
                else:
                    raise LanguageModelFileFormatException('%d-gram part: %d ngrams read, expecting empty line' % (n,count))
            else:
                

                match = re.match(template,line)
                if not match:
                    raise LanguageModelFileFormatException('%d-gram part: line %s does not match template %s' % (n,line,template))
                # add to LM tree
                (score,sequence,backoff) = match.groups()
                print('READ %s - %s' % (line,str((score,sequence,backoff))))
                float_score = float(score)
                if backoff is not None:
                    float_backoff = float(backoff)

                seqData = sequence.split()
#                 if self.tokenMapper is not None:
#                     seqData = map(self.tokenMapper,seqData)

                parentNode = self.lmTree.traverse(seqData[0:-1])
                newNode = LanguageModelTreeNode(seqData[-1],float_score,float_backoff)
                parentNode.addChild(newNode)
            count += 1


    # compute the probability of a sequence "a b c" (c is the current word, "a b" is the history)
    # see above for the rule
    def computeProbability(self, seq):
        cdef int startPos
        cdef float highestBackoff
        cdef LanguageModelTreeNode highestBackoffNode
        cdef LanguageModelTreeNode lastNode
        cdef int depth

        assert len(seq) >= 1
        seq = seq[max(len(seq) - self.maxN,0):len(seq)]

        # if this is more than an unigram, first find highest backoff

        if len(seq) == 1:
            highestBackoffNode = self.lmTree
            highestBackoff = 0.0
            startPos = 0
        else:
            for startPos in range(len(seq) - 1):
                try:
                    highestBackoffNode = self.lmTree.traverse(seq[startPos:-1])
                except LanguageModelStatsException:
                    pass
                else:
                    highestBackoff = highestBackoffNode.backoff
                    if highestBackoff is None:
                        pass
                    assert highestBackoff is not None
                    break
            else:
                raise LanguageModelStatsException('computeProbability: unigram %s not found' % seq[-1])

        # now check for maxN-gram
        try:
            lastNode = highestBackoffNode.children[seq[-1]]
            return lastNode.score
        except KeyError:
            # OK, so we have to back off with a suitable subsequence given by the highest backoff
            seq = seq[startPos:]
            accumulatedBackoff = highestBackoff
            oldSeq = seq[:] # copy
            while len(seq) > 0:
                # we don't find the n-gram for seq, of course
                assert self.lmTree.traversePartly(seq,0)[1] < len(seq)
                # so go one shorter
                seq = seq[1:]
                (lastNode,depth) = self.lmTree.traversePartly(seq)
                if depth < len(seq) - 1:
                    # because of the previous search for highestBackoff, we don't expect this
                    raise LanguageModelStatsException('computeProbability: %s found in LM, but sub-sequence %s not found' % (oldSeq,seq))
                if depth == len(seq):
                    # found
                    assert lastNode.token is not None
                    try:
                        res = lastNode.score + accumulatedBackoff
                    except TypeError as e:
                        print('Hae? ',e)
                    return res
                else:
                    # no result, but more backoff, at least
                    accumulatedBackoff += lastNode.backoff
            # since all unigrams should be present in the LM, we *cannot* be here
            raise LanguageModelStatsException('computeProbability: you should not be here. Your language model is shit')



#         # iterate over different backoffs
#         for startPos in range(0,len(seq)):
#             res = self.lmTree.getScoreOrBackoff(seq)
#             if res is not None:
#                 return res
#         assert False # should not reach this - TODO should be a true exception

#     # traverse the tree recursively along the history
#     # assumes that all unigrams have a score (TODO?)
#     # history is first ... last, and if this is a normal node, the 
#     # "current" token (self.token) is not included in the history
#     def traverseTree(self,history):
#         if self.token is None:
#             # starting
#             thisUnigramNode = self.children[history[-1]]
#             return thisUnigramNode.getScore(history[0:-1])
#         else:
#             try:
#                 nextNode = self.children[history[-1]]
#                 return nextNode.getScore(history[0:-1])
#             except KeyError:
#                 # back off
#                 return self.score 
# 
