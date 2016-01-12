from __future__ import division, print_function

import numpy as np
import pytest
import datetime
import operator

from brainstorm.languagemodel import LanguageModel
from brainstorm.layers.search import SearchTree

# assures that there are no loops in the tree
def check_tree_is_acyclic(search_tree):

    # recursive part
    def check_acyclicity_recursively(search_tree,start_node_index,visited_nodes):
        assert start_node_index not in visited_nodes
        visited_nodes.append(start_node_index)
        this_node = search_tree.nodes[start_node_index]
        for child_index in this_node.child_nodes.values():
            check_acyclicity_recursively(search_tree,child_index,visited_nodes)

    visited_nodes = [] # mutable
    for root_node_index in search_tree.root_nodes.values():
        check_acyclicity_recursively(search_tree,root_node_index,visited_nodes)

# takes a tree structure as nested list of tuples, checks the search tree for correctness
# example for list o a tree containing 123,124,34,345: [ (1,[ (2, [ (3, [], 'word1'), (4, [], 'word2') ]) ]), (3, [ (4, [ (5,[],'word4') ], 'word3') ] ) ]
# multiple words are allowed
def check_tree(search_tree,structure_list):

    # check a subtree, returns the number of nodes in this subtree
    # sub_structure must be a tuple (atom_id, children)
    def check_subtree(search_tree,start_node,sub_structure):
        this_atom = sub_structure[0]
        these_children = sub_structure[1]
        these_words = sub_structure[2:] if len(sub_structure) > 2 else None

        assert start_node.atom_id == this_atom
        if these_words is not None:
            assert set(start_node.tokens) == set(these_words)
        else:
            assert len(start_node.tokens) == 0
        assert len(start_node.child_nodes) == len(these_children)
        
        count = 1 # this node
        for child_structure in these_children:
            assert child_structure[0] in start_node.child_nodes.keys()
            count += check_subtree(search_tree,search_tree.nodes[start_node.child_nodes[child_structure[0]]],child_structure)

        return count

    # main part
    assert len(search_tree.root_nodes) == len(structure_list)
    assert len(set(search_tree.root_nodes)) == len(structure_list) # ...unique

    for root_node_structure in structure_list:
#         this_atom = root_node_structure[0]
#         these_children = root_node_structure[1]
#         these_words = root_node_structure[2:] if len(root_node_structure) > 2 else None
        # find root node with this atom, may just be one!
        root_node_index = search_tree.root_nodes[root_node_structure[0]]
        root_node = search_tree.nodes[root_node_index]
        assert root_node.atom_id == root_node_structure[0] 
        check_subtree(search_tree,root_node,root_node_structure)
        
        

def test_search_tree_builds_correctly():
    # may contain zeros, but if CTC is used, this creates a conflict
    phone_list = [ 0, 2, 4, 6, 8 ]
    vocabulary = [ 'ham', 'spam', 'eggs', 'sauce', 'mushrooms', 'spices' ]
    dictionary = { 'ham': [ 0, 2, 8 ], 'spam': [ 4, 6, 8 ], 'eggs': [ 4, 6, 0, 0 ], 'sauce': [ 0, 6 ], 'mushrooms': [ 6, 0], 'spices': [ 6, 0, 0, 4, 8 ] }
# # # # #     vocabulary = [ 'ham', 'spam', 'eggs', 'sauce', 'spices' ]
# # # # #     dictionary = { 'ham': [ 0, 2, 8 ], 'spam': [ 4, 6, 8 ], 'eggs': [ 4, 6, 0, 0 ], 'sauce': [ 0, 6 ], 'spices': [ 6, 0, 0, 4, 8 ] }

    # one word
    search_tree = SearchTree(phone_list,vocabulary[0:1],dictionary,None)
    check_tree_is_acyclic(search_tree)
    assert len(search_tree.root_nodes) == 1
    assert len(search_tree.final_nodes) == 1
    assert len(search_tree.nodes) == 3
    # check nodes and connections
    check_tree(search_tree,[(0,[(2,[(8,[],'ham')])])])
    assert search_tree.nodes[search_tree.final_nodes[0]].atom_id == 8
    assert search_tree.nodes[search_tree.final_nodes[0]].tokens == [ 'ham' ]

    # two words
    search_tree = SearchTree(phone_list,vocabulary[0:2],dictionary,None)
    check_tree_is_acyclic(search_tree)
    assert len(search_tree.root_nodes) == 2
    assert len(search_tree.final_nodes) == 2
    assert len(search_tree.nodes) == 6
    # check nodes and connections
    check_tree(search_tree,[(0,[(2,[(8,[],'ham')])]), (4,[(6,[(8,[],'spam')])]) ])
#     assert search_tree.nodes[search_tree.final_nodes[0]].atom_id == 8
#     assert search_tree.nodes[search_tree.final_nodes[0]].tokens == [ 'ham' ]
# 

    # all words
    search_tree = SearchTree(phone_list,vocabulary,dictionary,None)
    check_tree_is_acyclic(search_tree)
    assert len(search_tree.root_nodes) == 3
    assert len(search_tree.final_nodes) == len(dictionary)
    assert len(search_tree.nodes) == 14

    check_tree(search_tree,[(0,[(6,[],'sauce'),(2,[(8,[],'ham')])]), (4,[(6,[(0,[(0,[],'eggs')]),(8,[],'spam')])]), \
# # # # #             (6,[(0,[(0,[(4,[(8,[],'spices')])])])])         ])
            (6,[(0,[(0,[(4,[(8,[],'spices')])])],'mushrooms')])         ])
    
# Trace a path through the tree, returning the node corresponding to the last atom of the path
# if start_node_index is -1, start with the root nodes
def trace_path(search_tree,atoms,start_node_index = -1):
    if len(atoms) == 0:
        return (start_node_index,search_tree.nodes[start_node_index])
    # recurse
    this_dict = search_tree.nodes[start_node_index].child_nodes if start_node_index >= 0 else search_tree.root_nodes
#     assert atoms[0] in this_dict.keys()
    next_node_index = this_dict[atoms[0]]
    return trace_path(search_tree,atoms[1:],next_node_index)


# Check that a specific set of tree nodes has the expected hypos ('current' or 'next')
# uses np.isclose for score checks
# the expected_hypos is a dictionary (node index => [(score, history)] (a list))
def check_hypos(search_tree,expected_hypos,what):
    assert what in [ 'current', 'next', 'final' ]
    for (node_index,these_expected) in expected_hypos.items():
        if what == 'current':
            these_hypos = search_tree.nodes[node_index].current_hypos 
        elif what == 'next':
            these_hypos = search_tree.nodes[node_index].next_hypos 
        elif what == 'final':
            these_hypos = search_tree.nodes[node_index].current_final_hypos 
        # sort by score
        sorted_hypos = sorted(these_hypos,key=lambda h: h.score)
        sorted_expected = sorted(these_expected,key=operator.itemgetter(0))

        assert len(sorted_hypos) == len(sorted_expected)
        for pos in range(len(sorted_hypos)):
            assert np.isclose(sorted_hypos[pos].score,sorted_expected[pos][0])
            assert sorted_hypos[pos].history == sorted_expected[pos][1]


def test_internal_propagation_works():
    # build example tree
    phone_list = [ 1, 2, 4, 6, 8 ]
    vocabulary = [ 'ham', 'spam', 'eggs', 'pepper' ]
    dictionary = { 'ham': [ 1, 2, 8 ], 'spam': [ 4, 6, 8 ], 'eggs': [ 4, 6, 1, 1 ], 'pepper': [ 4, 6 ] }

    search_tree = SearchTree(phone_list,vocabulary,dictionary,None)

    # add some hypos
    ham_start_node_index,ham_start_node = trace_path(search_tree,[1])
    pre_branching_node_index,pre_branching_node = trace_path(search_tree,[4] )
    branching_node_index,branching_node = trace_path(search_tree,[4,6])
    eggs_end_index,eggs_end = trace_path(search_tree,[4,6,1,1])

    assert pre_branching_node.child_nodes[6] == branching_node_index

    search_tree.make_hypo(3.4,('spam',),ham_start_node_index,'current')
    search_tree.make_hypo(5.6,('eggs','ham'),ham_start_node_index,'current')

    search_tree.make_hypo(7.7,('eggs','ham'),pre_branching_node_index,'current')

    search_tree.make_hypo(9.43,('spam',),branching_node_index,'current')
    search_tree.make_hypo(4.93,('eggs','pepper'),branching_node_index,'current')

    search_tree.make_hypo(1.11,('spam',),eggs_end_index,'current')

    # Propagate and check
    neglog = np.random.rand(10).astype(np.double)
    search_tree.propagate(neglog)
    
    expected_hypos = {
            # self-loop for ham_start_node
            ham_start_node_index: [(3.4 + neglog[0],('spam',)),(5.6 + neglog[0],('eggs','ham'))],
            # one child for ham_start, atom 2
            ham_start_node.child_nodes[2]: [(3.4 + neglog[2],('spam',)),(5.6 + neglog[2],('eggs','ham'))],
            # self loop for pre_branching_node
            pre_branching_node_index: [(7.7 + neglog[0],('eggs','ham'))],
            # branching node, the case of pepper (finishes here) is unhandled, selfloop plus stuff from previous node
            branching_node_index: [(7.7 + neglog[6],('eggs','ham')),(9.43 + neglog[0],('spam',)),(4.93 + neglog[0],('eggs','pepper'))],
            # two children for branching node, the case of pepper (finishes here) is unhandled
            branching_node.child_nodes[1]: [(9.43 + neglog[1],('spam',)),(4.93 + neglog[1],('eggs','pepper'))],
            branching_node.child_nodes[8]: [(9.43 + neglog[8],('spam',)),(4.93 + neglog[8],('eggs','pepper'))],
            # self loop for eggs_end
            eggs_end_index: [(1.11 + neglog[0],('spam',))],
            }
           
    check_hypos(search_tree,expected_hypos,'next')

def test_external_propagation_works():
    # build example tree
    phone_list = [ 1, 2, 4, 6, 8 ]
    vocabulary = [ 'ham', 'spam', 'eggs', 'pepper' ]
    dictionary = { 'ham': [ 1, 2, 8 ], 'spam': [ 4, 6, 8 ], 'eggs': [ 4, 6, 1, 1 ], 'pepper': [ 4, 6 ] }

    search_tree = SearchTree(phone_list,vocabulary,dictionary,None)

    # add some FINAL hypos
    ham_end_index,ham_end = trace_path(search_tree,[1,2,8])
    spam_end_index,spam_end = trace_path(search_tree,[4,6,8])
    eggs_end_index,eggs_end = trace_path(search_tree,[4,6,1,1])
    pepper_end_index,pepper_end = trace_path(search_tree,[4,6])

    search_tree.make_hypo(3.4,('spam','ham'),ham_end_index,'final')
    search_tree.make_hypo(1.9,('eggs','ham'),ham_end_index,'final')

    search_tree.make_hypo(2.22,('spam',),spam_end_index,'final')

    search_tree.make_hypo(5,('spam','ham','eggs'),eggs_end_index,'final')
    search_tree.make_hypo(5.1,('eggs','eggs',),eggs_end_index,'final')

    # none for pepper (kind of a corner case)

    # Propagate and check
    neglog = np.random.rand(10).astype(np.double)
    search_tree.propagate_external(neglog)
    
    expected_hypos = {
            # root node 1
            search_tree.root_nodes[1]: [ (3.4 + neglog[1],('spam','ham')), (1.9 + neglog[1],('eggs','ham')),
                (2.22 + neglog[1],('spam',)), (5 + neglog[1],('spam','ham','eggs')), (5.1 + neglog[1],('eggs','eggs',)) ],
            search_tree.root_nodes[4]: [ (3.4 + neglog[4],('spam','ham')), (1.9 + neglog[4],('eggs','ham')),
                (2.22 + neglog[4],('spam',)), (5 + neglog[4],('spam','ham','eggs')), (5.1 + neglog[4],('eggs','eggs',)) ]
            }
           
    check_hypos(search_tree,expected_hypos,'next')



def test_finalization_works():
    # this means that nodes are copied correctly from 'next' to 'current' and 'final' hypos

    def add_some_hypos(search_tree,language_model = None):
        # see below, e.g. call_lm((1.1,('bla','bla'))) returns the same tuple, but the 
        # LM score is added to the score as given
        if language_model is None:
            call_lm = lambda tup: tup
        else:
            call_lm = lambda tup: (tup[0] + language_model.computeProbability(tup[1]),tup[1])

        # add some NORMAL ('next') hypos which should be converted into current and possibly final ones
        ham_start_index,ham_start_node = trace_path(search_tree,[1])
        ham_end_index,ham_end = trace_path(search_tree,[1,2,8])
        spam_end_index,spam_end = trace_path(search_tree,[4,6,8])
        eggs_end_index,eggs_end = trace_path(search_tree,[4,6,1,1])
        pepper_end_index,pepper_end = trace_path(search_tree,[4,6])

        search_tree.make_hypo(0.1,('pepper','ham'),ham_start_index,'next')

        search_tree.make_hypo(3.4,('spam','ham'),ham_end_index,'next')
        search_tree.make_hypo(1.9,('eggs',),ham_end_index,'next')

        search_tree.make_hypo(2.2,('ham',),spam_end_index,'next')

        search_tree.make_hypo(5,('spam','ham','eggs'),eggs_end_index,'next')
        search_tree.make_hypo(5.1,('eggs','eggs',),eggs_end_index,'next')

        search_tree.finish_propagation()

        expected_current_hypos = {
                ham_start_index: [(0.1,('pepper','ham'))],
                ham_end_index: [(3.4,('spam','ham')),(1.9,('eggs',))],
                spam_end_index: [(2.2,('ham',))],
                eggs_end_index: [(5,('spam','ham','eggs')),(5.1,('eggs','eggs'))],
                }

        expected_final_hypos = {
                ham_end_index: [call_lm((3.4,('spam','ham','ham'))),call_lm((1.9,('eggs','ham'))),call_lm((3.4,('spam','ham','bacon'))),call_lm((1.9,('eggs','bacon')))],
                spam_end_index: [call_lm((2.2,('ham','spam')))],
                eggs_end_index: [call_lm((5,('spam','ham','eggs','eggs'))),call_lm((5.1,('eggs','eggs','eggs')))],
                }

        return (expected_current_hypos,expected_final_hypos)
     
    phone_list = [ 1, 2, 4, 6, 8 ]
    vocabulary = [ 'ham', 'spam', 'eggs', 'pepper', 'bacon' ]
    # two tokens with identical atom lists
    dictionary = { 'ham': [ 1, 2, 8 ], 'spam': [ 4, 6, 8 ], 'eggs': [ 4, 6, 1, 1 ], 'pepper': [ 4, 6 ], 'bacon': [ 1,2,8 ] }

    # WITHOUT language model
    search_tree = SearchTree(phone_list,vocabulary,dictionary,None)

    expected_current,expected_final = add_some_hypos(search_tree)

    check_hypos(search_tree,expected_current,'current')
    check_hypos(search_tree,expected_final,'final')

    # WITH language model
    language_model_data = r"""
\data\
ngram 1=5
ngram 2=9

\1-grams:
-1.86   ham     -2.40
-3.59   spam    -2.83
-1.79   eggs    -2.79
-2.11   pepper  -1.22
-2.55   bacon   -1.90

\2-grams:
-0.56            ham ham
-4.62            ham bacon
-5.55            eggs eggs
-4.44            spam eggs
-9.44            eggs ham
-3.11            bacon eggs
-0.97            pepper eggs
-0.02            ham pepper
-2.11            pepper pepper

\end\
"""
    
    assert isinstance(language_model_data,basestring)
    lm = LanguageModel(language_model_data)

    search_tree = SearchTree(phone_list,vocabulary,dictionary,lm)

    expected_current,expected_final = add_some_hypos(search_tree,lm)

    check_hypos(search_tree,expected_current,'current')
    check_hypos(search_tree,expected_final,'final')




