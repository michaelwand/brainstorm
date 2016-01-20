from __future__ import division, print_function

import numpy as np
import pytest
import datetime
import operator
import pickle
import pdb

from brainstorm.languagemodel import LanguageModel
from brainstorm.layers.search import SearchTree, HypoContainer

################# TESTS FOR STATIC TREE #################

trivial_beams = { 'hypo_beam': np.inf, 'hypo_topn': 10000, 'final_beam': np.inf, 'final_topn': 10000, 'token_insertion_penalty': 0, 'lm_weight': 0 }
Dictionary = { 'ham': [[ 1, 2, 8 ]], 'spam': [[ 4, 6, 8 ]], 'eggs': [[ 4, 6, 1, 1 ]], 'sauce': [[ 1, 6 ]], 'mushrooms': [[ 6, 1]], 'spices': [[ 6, 1, 1, 4, 8 ]], 'pepper': [[ 4,6 ]] }

# # assures that there are no loops in the tree
# def check_tree_is_acyclic(search_tree):
#     return  1
# 
# #     prior_nodes = [] # mutable
# # 
# #     # recursive part
# #     def check_acyclicity_recursively(search_tree,start_node_index,prior_nodes):
# #         assert start_node_index not in prior_nodes
# #         prior_nodes.append(start_node_index)
# #         this_node = search_tree.nodes[start_node_index]
# #         for child_index in this_node.child_nodes.values():
# #             check_acyclicity_recursively(search_tree,child_index,prior_nodes)
# # 
# #     for root_node_index in search_tree.root_nodes.values():
# #         check_acyclicity_recursively(search_tree,root_node_index,prior_nodes)

def check_tree(search_tree,vertices,edges,root_nodes):

    # Check a subgraph, starting from a node in search_tree with an id,
    # and the corresponding index in the passed-in description. Accesses
    # nonlocal list of visited search_tree nodes to avoid covering any part twice.

    visited_tree_node_indices = []

    def check_subtree(tree_node_index,vertex_index):
        tree_node = search_tree.nodes[tree_node_index]
        visited_tree_node_indices.append(tree_node_index)
        # first check node properties
        assert tree_node.atom_id == vertices[vertex_index][0]
        assert set(tree_node.tokens) == set(vertices[vertex_index][1])
        if len(tree_node.tokens) > 0:
            assert tree_node_index in search_tree.final_nodes
        assert len(tree_node.child_nodes) == len(edges[vertex_index]) 

        # now recurse along connections, auto-stops if there are no children
        child_vertices = edges[vertex_index]
        for child_vertex in child_vertices:
            child_atom = vertices[child_vertex][0]
            assert child_atom in tree_node.child_nodes.keys()
            child_tree_node_index = tree_node.child_nodes[child_atom]
            if child_tree_node_index not in visited_tree_node_indices:
                check_subtree(child_tree_node_index,child_vertex)

    ### MAIN PART OF check_tree ###

    assert len(set(search_tree.root_nodes)) == len(search_tree.root_nodes) # ...unique
    assert len(search_tree.root_nodes) == len(root_nodes)
    assert len(search_tree.nodes) == len(vertices)
    
    # Main part. For each root node (distinguished by atom), follow...
    for root_node in root_nodes:
        root_node_atom = vertices[root_node][0]
#         this_atom = root_node_structure[0]
#         these_children = root_node_structure[1]
#         these_words = root_node_structure[2:] if len(root_node_structure) > 2 else None
        # find root node with this atom, may just be one!
        tree_root_node_index = search_tree.root_nodes[root_node_atom]
#         tree_root_node = search_tree.nodes[root_node_index]
#         assert root_node.atom_id == root_node_structure[0] 
        check_subtree(tree_root_node_index,root_node)
      
# Merge dictionaries by collecting values from the source dictionaries
# into a list and processing it by proc_values
def merge_dicts(dict_list,proc_values):
    for d in dict_list:
        assert isinstance(d,dict)
    all_keys = [ d.keys() for d in dict_list ]
    all_key_set = set([])
    all_key_set.update(*all_keys)

    result_dict = {}
    for k in all_key_set:
        values = [ d[k] for d in dict_list if k in d.keys() ]
        result_dict[k] = proc_values(values)

    return result_dict

def merge_vertex_dicts(dict_list):
    def proc_vertices(lst):
        for pos in range(1,len(lst)):
            assert lst[pos] == lst[0]
        return lst[0]

    return merge_dicts(dict_list,proc_vertices)

def merge_edge_dicts(dict_list):
    def proc_edges(edge_tuple_lst):
        result_edge_set = set([])
        result_edge_set.update(*edge_tuple_lst)
        return tuple(result_edge_set)

    return merge_dicts(dict_list,proc_edges)
#     for d in dict_list:
#         assert isinstance(d,dict)
#     all_keys = [ d.keys() for d in dict_list ]
#     all_key_set = set([])
#     all_key_set.update(*all_keys)
# 
#     result_dict = {}
#     for k in all_key_set:
#         # collect all edges 
#         all_edges = []
#         for d in dict_list:
#             if k in d.keys():
#                 all_edges.extend(d[k]) 
#         result_dict[k] = tuple(all_edges)
#     return result_dict

def test_search_tree_builds_correctly():
    # may contain zeros, but if CTC is used, this creates a conflict
    phone_list = [ 0, 2, 4, 6, 8 ]
    vocabulary = [ 'ham', 'spam', 'eggs', 'sauce', 'mushrooms', 'spices' ]

    # vertices: index -> atom, tokens; edges: index to tuple(!) of indices, root nodes: a list or iterable
    ham_vertices = { 0: (0,()), 1: (1,()), 2: (0,()), 3: (2,()), 4: (0,()), 5: (8,('ham',)), 6: (0,('ham',)) }
    ham_edges = { 0: (1,), 1: (2,3), 2: (3,), 3: (4,5), 4: (5,), 5: (6,), 6: () }
    ham_root_nodes = set([0,1])

    spam_vertices = { 0: (0,()), 
            11: (4,()),
            12: (0,()),
            13: (6,()),
            14: (0,()),
            15: (8,('spam',)),
            16: (0,('spam',)) }

    spam_edges = { 0: (11,), 11: (12,13), 12: (13,), 13: (14,15), 14: (15,), 15: (16,), 16: () }
    spam_root_nodes = set([0,11])
    
    eggs_vertices = { 0: (0,()), 
            11: (4,()),
            12: (0,()),
            13: (6,()),
            14: (0,()),
            21: (1,()),
            22: (0,()),
            23: (1,('eggs',)),
            24: (0,('eggs',)) }

    eggs_edges = { 0: (11,), 11: (12,13), 12: (13,), 13: (14,21), 14: (21,), 21: (22,), 22: (23,), 23: (24,), 24: () }
    eggs_root_nodes = set([0,11])
    
    sauce_vertices = { 0: (0,()), 1: (1,()), 2: (0,()), 31: (6,('sauce',)), 32: (0,('sauce',)) }
    sauce_edges = { 0: (1,), 1: (2,31), 2: (31,), 31: (32,), 32: () }
    sauce_root_nodes = set([0,1])

    mushrooms_vertices = { 0: (0,()), 
            41: (6,()),
            42: (0,()),
            43: (1,('mushrooms',)),
            44: (0,('mushrooms',)) }
    mushrooms_edges = { 0: (41,), 41: (42,43), 42: (43,), 43: (44,), 44: () }
    mushrooms_root_nodes = set([0,41])
    
    spices_vertices = { 0: (0,()), 
            41: (6,()),
            42: (0,()),
            43: (1,()),
            44: (0,()), 
            45: (1,()), 
            46: (0,()), 
            47: (4,()), 
            48: (0,()), 
            49: (8,('spices',)), 
            50: (0,('spices',)), 
            }
    spices_edges = { 0: (41,), 
            41: (42,43), 
            42: (43,), 
            43: (44,),
            44: (45,),
            45: (46,47), 
            46: (47,),
            47: (48,49), 
            48: (49,),
            49: (50,), 
            50: () }
    spices_root_nodes = set([0,41])
    
    m_s_vertices = { 0: (0,()), 
            41: (6,()),
            42: (0,()),
            43: (1,('mushrooms',)),
            44: (0,('mushrooms',)), 
            45: (1,()), 
            46: (0,()), 
            47: (4,()), 
            48: (0,()), 
            49: (8,('spices',)), 
            50: (0,('spices',)), 
            }
    m_s_edges = { 0: (41,), 
            41: (42,43), 
            42: (43,), 
            43: (44,),
            44: (45,),
            45: (46,47), 
            46: (47,),
            47: (48,49), 
            48: (49,),
            49: (50,), 
            50: () }
    m_s_root_nodes = set([0,41])
    


    # one word
    search_tree = SearchTree(phone_list,vocabulary[0:1],Dictionary,trivial_beams,None)
    # check nodes and connections
    check_tree(search_tree,ham_vertices,ham_edges,ham_root_nodes)


    # two words
    search_tree = SearchTree(phone_list,vocabulary[0:2],Dictionary,trivial_beams,None)
    # check nodes and connections
    check_tree(search_tree,merge_vertex_dicts([ham_vertices,spam_vertices]),merge_edge_dicts([ham_edges,spam_edges]),ham_root_nodes | spam_root_nodes)

    # almost all words
    search_tree = SearchTree(phone_list,vocabulary[0:5],Dictionary,trivial_beams,None)
    almost_all_vertices = merge_vertex_dicts([ham_vertices,spam_vertices,eggs_vertices,sauce_vertices,mushrooms_vertices])
    almost_all_edges = merge_edge_dicts([ham_edges,spam_edges,eggs_edges,sauce_edges,mushrooms_edges])
    almost_all_root_nodes = ham_root_nodes | spam_root_nodes | eggs_root_nodes | sauce_root_nodes | mushrooms_root_nodes
    check_tree(search_tree,almost_all_vertices,almost_all_edges,almost_all_root_nodes)

    # same same but different
    search_tree = SearchTree(phone_list, [ 'ham', 'spam', 'eggs', 'sauce', 'spices' ],Dictionary,trivial_beams,None)
    almost_all_vertices = merge_vertex_dicts([ham_vertices,spam_vertices,eggs_vertices,sauce_vertices,spices_vertices])
    almost_all_edges = merge_edge_dicts([ham_edges,spam_edges,eggs_edges,sauce_edges,spices_edges])
    almost_all_root_nodes = ham_root_nodes | spam_root_nodes | eggs_root_nodes | sauce_root_nodes | spices_root_nodes
    check_tree(search_tree,almost_all_vertices,almost_all_edges,almost_all_root_nodes)

    # really all
    search_tree = SearchTree(phone_list, vocabulary,Dictionary,trivial_beams,None)
    all_vertices = merge_vertex_dicts([ham_vertices,spam_vertices,eggs_vertices,sauce_vertices,m_s_vertices])
    all_edges = merge_edge_dicts([ham_edges,spam_edges,eggs_edges,sauce_edges,m_s_edges])
    all_root_nodes = ham_root_nodes | spam_root_nodes | eggs_root_nodes | sauce_root_nodes | m_s_root_nodes
    check_tree(search_tree,all_vertices,all_edges,all_root_nodes)

    # extra alternative
    extended_dict = Dictionary.copy()
    extended_dict['ham'] = [ [ 1,2,8], [1,4,4 ] ]
    ext_ham_vertices = { 0: (0,()), 1: (1,()), 2: (0,()), 3: (2,()), 4: (0,()), 5: (8,('ham',)), 
            6: (0,('ham',)), 7: (4,()), 8: (0,()), 9: (4,('ham',)), 10: (0,('ham',))  }
    ext_ham_edges = { 0: (1,), 1: (2,3,7), 2: (3,7), 3: (4,5), 4: (5,), 5: (6,), 6: (), 7: (8,), 8: (9,), 9: (10,), 10: () }
    ext_ham_root_nodes = set([0,1])

    # only ham
    search_tree = SearchTree(phone_list, ['ham'],extended_dict,trivial_beams,None)
    check_tree(search_tree,ext_ham_vertices,ext_ham_edges,ext_ham_root_nodes)
   
    # everything
    search_tree = SearchTree(phone_list, vocabulary,extended_dict,trivial_beams,None)
    all_vertices = merge_vertex_dicts([ext_ham_vertices,spam_vertices,eggs_vertices,sauce_vertices,m_s_vertices])
    all_edges = merge_edge_dicts([ext_ham_edges,spam_edges,eggs_edges,sauce_edges,m_s_edges])
    all_root_nodes = ext_ham_root_nodes | spam_root_nodes | eggs_root_nodes | sauce_root_nodes | m_s_root_nodes
    check_tree(search_tree,all_vertices,all_edges,all_root_nodes)
    
################# TESTS FOR DYNAMIC ALGORITHMS #################

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

# check that a HypoContainer has the expected hypos
# uses np.isclose for score checks
# the expected_hypos is a list [(score, history)] 
def check_hypos(hypo_container,expected_hypos):
        # sort by score
        sorted_hypos = sorted(hypo_container.p_get_hypo_iterator(),key=operator.itemgetter(0))
        sorted_expected = sorted(expected_hypos,key=operator.itemgetter(0))
        
        assert len(sorted_hypos) == len(sorted_expected)
        for pos in range(len(sorted_hypos)):
            assert np.isclose(sorted_hypos[pos][0],sorted_expected[pos][0])
            assert sorted_hypos[pos][1] == sorted_expected[pos][1]

# Check that a specific set of tree nodes has the expected hypos ('current' or 'next' or 'final')
# the expected_hypos is a dictionary (node index => [(score, history)] (a list))
def check_tree_hypos(search_tree,expected_hypos,what):
    assert what in [ 'current', 'next', 'final' ]
    unmentioned_nodes = set(range(len(search_tree.nodes))) - set(expected_hypos.keys())
    for (node_index,these_expected) in expected_hypos.items():
        if what == 'current':
            this_hypo_container = search_tree.nodes[node_index].current_hypos
        elif what == 'next':
            this_hypo_container = search_tree.nodes[node_index].next_hypos
        elif what == 'final':
            this_hypo_container = search_tree.nodes[node_index].current_final_hypos

        check_hypos(this_hypo_container,these_expected)

    for node_index in unmentioned_nodes:
        if what == 'current':
            this_hypo_container = search_tree.nodes[node_index].current_hypos
        elif what == 'next':
            this_hypo_container = search_tree.nodes[node_index].next_hypos
        elif what == 'final':
            this_hypo_container = search_tree.nodes[node_index].current_final_hypos

        check_hypos(this_hypo_container,[])



# Check that the HypoContainer class works
def test_hypo_container_insertion_and_clearing():
    cont = HypoContainer()
    check_hypos(cont,{})
    cont.p_insert(2.1,('a','b'))
    check_hypos(cont,[(2.1,('a','b'))])
    cont.p_insert(3.4,('b','c'))
    check_hypos(cont,[(2.1,('a','b')),(3.4,('b','c'))])
    cont.p_insert(5.6,('a','b')) # should have no effect
    check_hypos(cont,[(2.1,('a','b')),(3.4,('b','c'))])
    cont.p_insert(0.1,('b','c')) # replace previous one
    check_hypos(cont,[(2.1,('a','b')),(0.1,('b','c'))])
    cont.p_clear()
    check_hypos(cont,{})

# Check that the HypoContainer class works
def test_hypo_container_pruning():
    cont = HypoContainer()
    cont.p_insert(2.1,('a',))
    cont.p_insert(3.4,('b',))
    cont.p_insert(5.6,('c',))
    cont.p_insert(0.1,('d',))
    cont.p_insert(2.9,('e',))
    cont.p_insert(0.3,('f','g'))
    
    cont.p_keep_nbest(7) # nothing happens
    check_hypos(cont,[(2.1,('a',)), (3.4,('b',)), (5.6,('c',)), (0.1,('d',)), (2.9,('e',)), (0.3,('f','g'))])
    cont.p_keep_nbest(6) # nothing happens
    check_hypos(cont,[(2.1,('a',)), (3.4,('b',)), (5.6,('c',)), (0.1,('d',)), (2.9,('e',)), (0.3,('f','g'))])
    cont.p_keep_nbest(5) 
    check_hypos(cont,[(2.1,('a',)), (3.4,('b',)), (0.1,('d',)), (2.9,('e',)), (0.3,('f','g'))])
    cont.p_keep_nbest(2) 
    check_hypos(cont,[(0.1,('d',)), (0.3,('f','g'))])


def test_internal_propagation_works():
    # build example tree
    phone_list = [ 1, 2, 4, 6, 8 ]
    vocabulary = [ 'ham', 'spam', 'eggs', 'pepper' ]
#     dictionary = { 'ham': [ 1, 2, 8 ], 'spam': [ 4, 6, 8 ], 'eggs': [ 4, 6, 1, 1 ], 'pepper': [ 4, 6 ] }

    search_tree = SearchTree(phone_list,vocabulary,Dictionary,trivial_beams,None)

    # add some hypos
    search_tree.clear_hypos()
    ham_start_node_index,ham_start_node = trace_path(search_tree,[1])
    ham_blank_node_index,ham_blank_node = trace_path(search_tree,[1,0,2,0])
    pre_branching_node_index,pre_branching_node = trace_path(search_tree,[4] )
    branching_node_index,branching_node = trace_path(search_tree,[4,6])
    alt_branching_node_index,alt_branching_node = trace_path(search_tree,[4,0,6])
    # this does really not belong here, TODO remove
    assert branching_node_index == alt_branching_node_index

    eggs_end_index,eggs_end = trace_path(search_tree,[4,6,1,0,1])

    assert pre_branching_node.child_nodes[6] == branching_node_index

    search_tree.make_hypo(3.4,('spam',),ham_start_node_index,'current')
    search_tree.make_hypo(5.6,('eggs','ham'),ham_start_node_index,'current')

    search_tree.make_hypo(2.22,('spam','ham'),ham_blank_node_index,'current')

    search_tree.make_hypo(7.7,('eggs','ham'),pre_branching_node_index,'current')

    search_tree.make_hypo(9.43,('spam',),branching_node_index,'current')
    search_tree.make_hypo(4.93,('eggs','pepper'),branching_node_index,'current')

    search_tree.make_hypo(1.11,('spam',),eggs_end_index,'current')

    # Propagate and check
    neglog = np.random.rand(10) #.astype(double)
    search_tree.propagate(neglog)
    
    expected_hypos = {
            # self-loop for ham_start_node
            ham_start_node_index: [(3.4 + neglog[1],('spam',)),(5.6 + neglog[1],('eggs','ham'))],
            # children for ham_start, atoms 0, 2
            ham_start_node.child_nodes[0]: [(3.4 + neglog[0],('spam',)),(5.6 + neglog[0],('eggs','ham'))],
            ham_start_node.child_nodes[2]: [(3.4 + neglog[2],('spam',)),(5.6 + neglog[2],('eggs','ham'))],

            # one blank node, let's see
            ham_blank_node_index: [(2.22 + neglog[0],('spam','ham'))],
            ham_blank_node.child_nodes[8]: [(2.22 + neglog[8],('spam','ham'))],
            
            # pre-branching node
            pre_branching_node_index: [(7.7 + neglog[4],('eggs','ham'))],
            pre_branching_node.child_nodes[0]: [(7.7 + neglog[0],('eggs','ham'))],

            # branching node, the case of pepper (finishes here) is unhandled, selfloop plus stuff from previous node
            branching_node_index: [(7.7 + neglog[6],('eggs','ham')),(9.43 + neglog[6],('spam',)),(4.93 + neglog[6],('eggs','pepper'))],
            # three children for branching node, the case of pepper (finishes here) is unhandled
            branching_node.child_nodes[0]: [(9.43 + neglog[0],('spam',)),(4.93 + neglog[0],('eggs','pepper'))],
            branching_node.child_nodes[1]: [(9.43 + neglog[1],('spam',)),(4.93 + neglog[1],('eggs','pepper'))],
            branching_node.child_nodes[8]: [(9.43 + neglog[8],('spam',)),(4.93 + neglog[8],('eggs','pepper'))],
            # self loop for eggs_end
            eggs_end_index: [(1.11 + neglog[1],('spam',))],
            eggs_end.child_nodes[0]: [(1.11 + neglog[0],('spam',))],
            }
           
    check_tree_hypos(search_tree,expected_hypos,'next')


def test_internal_propagation_works_with_beams():
    # build example tree
    phone_list = [ 1, 2, 4, 6, 8 ]
    vocabulary = [ 'ham', 'spam', 'eggs', 'pepper' ]
#     dictionary = { 'ham': [ 1, 2, 8 ], 'spam': [ 4, 6, 8 ], 'eggs': [ 4, 6, 1, 1 ], 'pepper': [ 4, 6 ] }

    search_tree = SearchTree(phone_list,vocabulary,Dictionary,{ 'hypo_beam': 5.0, 'hypo_topn': 10000, 'final_beam': 2.0, 'final_topn': 10000, 'token_insertion_penalty': 0, 'lm_weight': 0 },None)

    # add some hypos
    search_tree.clear_hypos()
    ham_start_node_index,ham_start_node = trace_path(search_tree,[1])
    ham_blank_node_index,ham_blank_node = trace_path(search_tree,[1,0,2,0])
    pre_branching_node_index,pre_branching_node = trace_path(search_tree,[4] )
    branching_node_index,branching_node = trace_path(search_tree,[4,6])
    eggs_end_index,eggs_end = trace_path(search_tree,[4,6,1,0,1])

    assert pre_branching_node.child_nodes[6] == branching_node_index

    search_tree.make_hypo(3.4,('spam',),ham_start_node_index,'current')
    search_tree.make_hypo(5.6,('eggs','ham'),ham_start_node_index,'current')

    search_tree.make_hypo(2.22,('spam','ham'),ham_blank_node_index,'current')

    search_tree.make_hypo(7.7,('eggs','ham'),pre_branching_node_index,'current')

    search_tree.make_hypo(9.43,('spam',),branching_node_index,'current')
    search_tree.make_hypo(4.93,('eggs','pepper'),branching_node_index,'current')

    search_tree.make_hypo(1.11,('spam',),eggs_end_index,'current')

    # prune tree just to get the pruning thresholds
    search_tree.prune_tree()

    # Propagate and check
    neglog = np.random.rand(10) #.astype(double)
    search_tree.propagate(neglog)
    
    expected_hypos = {
            # self-loop for ham_start_node
            ham_start_node_index: [(3.4 + neglog[1],('spam',)),(5.6 + neglog[1],('eggs','ham'))],
            # children for ham_start, atoms 0, 2
            ham_start_node.child_nodes[0]: [(3.4 + neglog[0],('spam',)),(5.6 + neglog[0],('eggs','ham'))],
            ham_start_node.child_nodes[2]: [(3.4 + neglog[2],('spam',)),(5.6 + neglog[2],('eggs','ham'))],

            # one blank node
            ham_blank_node_index: [(2.22 + neglog[0],('spam','ham'))],
            ham_blank_node.child_nodes[8]: [(2.22 + neglog[8],('spam','ham'))],
            
            # pre-branching node
            pre_branching_node_index: [],
            pre_branching_node.child_nodes[0]: [],

            # branching node, the case of pepper (finishes here) is unhandled, selfloop plus stuff from previous node
            branching_node_index: [(4.93 + neglog[6],('eggs','pepper'))],
            # three children for branching node, the case of pepper (finishes here) is unhandled
            branching_node.child_nodes[0]: [(4.93 + neglog[0],('eggs','pepper'))],
            branching_node.child_nodes[1]: [(4.93 + neglog[1],('eggs','pepper'))],
            branching_node.child_nodes[8]: [(4.93 + neglog[8],('eggs','pepper'))],
            # self loop for eggs_end
            eggs_end_index: [(1.11 + neglog[1],('spam',))],
            eggs_end.child_nodes[0]: [(1.11 + neglog[0],('spam',))],
            }
#     expected_hypos = {
#             # self-loop for ham_start_node
#             ham_start_node_index: [(3.4 + neglog[0],('spam',)),(5.6 + neglog[0],('eggs','ham'))],
#             # one child for ham_start, atom 2
#             ham_start_node.child_nodes[2]: [(3.4 + neglog[2],('spam',)),(5.6 + neglog[2],('eggs','ham'))],
#             # self loop for pre_branching_node
#             pre_branching_node_index: [],
#             # branching node, the case of pepper (finishes here) is unhandled, selfloop plus stuff from previous node
#             branching_node_index: [(4.93 + neglog[0],('eggs','pepper'))],
#             # two children for branching node, the case of pepper (finishes here) is unhandled
#             branching_node.child_nodes[1]: [(4.93 + neglog[1],('eggs','pepper'))],
#             branching_node.child_nodes[8]: [(4.93 + neglog[8],('eggs','pepper'))],
#             # self loop for eggs_end
#             eggs_end_index: [(1.11 + neglog[0],('spam',))],
#             }
           
    check_tree_hypos(search_tree,expected_hypos,'next')

# def test_internal_propagation_works_with_pruning():
#     # build example tree
#     phone_list = [ 1, 2, 4, 6, 8 ]
#     vocabulary = [ 'ham', 'spam', 'eggs', 'pepper' ]
#     dictionary = { 'ham': [ 1, 2, 8 ], 'spam': [ 4, 6, 8 ], 'eggs': [ 4, 6, 1, 1 ], 'pepper': [ 4, 6 ] }
# 
#     search_tree = SearchTree(phone_list,vocabulary,dictionary,None)
# 
#     # add some hypos
#     ham_start_node_index,ham_start_node = trace_path(search_tree,[1])
#     pre_branching_node_index,pre_branching_node = trace_path(search_tree,[4] )
#     branching_node_index,branching_node = trace_path(search_tree,[4,6])
#     eggs_end_index,eggs_end = trace_path(search_tree,[4,6,1,1])
# 
#     assert pre_branching_node.child_nodes[6] == branching_node_index
# 
#     search_tree.make_hypo(3.4,('spam',),ham_start_node_index,'current')
#     search_tree.make_hypo(5.6,('eggs','ham'),ham_start_node_index,'current')
# 
#     search_tree.make_hypo(7.7,('eggs','ham'),pre_branching_node_index,'current')
# 
#     search_tree.make_hypo(9.43,('spam',),branching_node_index,'current')
#     search_tree.make_hypo(4.93,('eggs','pepper'),branching_node_index,'current')
# 
#     search_tree.make_hypo(1.11,('spam',),eggs_end_index,'current')
# 
#     # Propagate and check
#     neglog = np.random.rand(10).astype(np.double)
#     search_tree.propagate(neglog)
#     
#     expected_hypos = {
#             # self-loop for ham_start_node
#             ham_start_node_index: [(3.4 + neglog[0],('spam',)),(5.6 + neglog[0],('eggs','ham'))],
#             # one child for ham_start, atom 2
#             ham_start_node.child_nodes[2]: [(3.4 + neglog[2],('spam',)),(5.6 + neglog[2],('eggs','ham'))],
#             # self loop for pre_branching_node
#             pre_branching_node_index: [(7.7 + neglog[0],('eggs','ham'))],
#             # branching node, the case of pepper (finishes here) is unhandled, selfloop plus stuff from previous node
#             branching_node_index: [(7.7 + neglog[6],('eggs','ham')),(9.43 + neglog[0],('spam',)),(4.93 + neglog[0],('eggs','pepper'))],
#             # two children for branching node, the case of pepper (finishes here) is unhandled
#             branching_node.child_nodes[1]: [(9.43 + neglog[1],('spam',)),(4.93 + neglog[1],('eggs','pepper'))],
#             branching_node.child_nodes[8]: [(9.43 + neglog[8],('spam',)),(4.93 + neglog[8],('eggs','pepper'))],
#             # self loop for eggs_end
#             eggs_end_index: [(1.11 + neglog[0],('spam',))],
#             }
#            
#     check_tree_hypos(search_tree,expected_hypos,'next')
def test_external_propagation_works():
    # build example tree
    phone_list = [ 1, 2, 4, 6, 8 ]
    vocabulary = [ 'ham', 'spam', 'eggs', 'pepper' ]
#     dictionary = { 'ham': [ 1, 2, 8 ], 'spam': [ 4, 6, 8 ], 'eggs': [ 4, 6, 1, 1 ], 'pepper': [ 4, 6 ] }

    search_tree = SearchTree(phone_list,vocabulary,Dictionary,trivial_beams,None)

    # add some FINAL hypos
    search_tree.clear_hypos()
    ham_end_index,ham_end = trace_path(search_tree,[1,2,8])
    spam_end_index,spam_end = trace_path(search_tree,[4,6,8])
    eggs_end_index,eggs_end = trace_path(search_tree,[4,6,1,0,1])
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
            search_tree.root_nodes[0]: [ (3.4 + neglog[0],('spam','ham')), (1.9 + neglog[0],('eggs','ham')),
                (2.22 + neglog[0],('spam',)), (5 + neglog[0],('spam','ham','eggs')), (5.1 + neglog[0],('eggs','eggs',)) ],
            search_tree.root_nodes[1]: [ (3.4 + neglog[1],('spam','ham')), (1.9 + neglog[1],('eggs','ham')),
                (2.22 + neglog[1],('spam',)), (5 + neglog[1],('spam','ham','eggs')), (5.1 + neglog[1],('eggs','eggs',)) ],
            search_tree.root_nodes[4]: [ (3.4 + neglog[4],('spam','ham')), (1.9 + neglog[4],('eggs','ham')),
                (2.22 + neglog[4],('spam',)), (5 + neglog[4],('spam','ham','eggs')), (5.1 + neglog[4],('eggs','eggs',)) ]
            }
           
    check_tree_hypos(search_tree,expected_hypos,'next')



def test_finalization_works():
    # this means that nodes are copied correctly from 'next' to 'current' and 'final' hypos

    def add_some_hypos(search_tree,language_model = None):
        # see below, e.g. call_lm((1.1,('bla','bla'))) returns the same tuple, but the 
        # LM score is added to the score as given
        if language_model is None:
            call_lm = lambda tup: tup
        else:
            call_lm = lambda tup: (tup[0] - language_model.computeProbability(tup[1]),tup[1])

        # add some NORMAL ('next') hypos which should be converted into current and possibly final ones
        search_tree.clear_hypos()
        ham_start_index,ham_start_node = trace_path(search_tree,[1])
        ham_end_index,ham_end = trace_path(search_tree,[1,2,8])
        spam_end_index,spam_end = trace_path(search_tree,[4,6,8])
        eggs_end_index,eggs_end = trace_path(search_tree,[4,6,1,0,1])
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
    # special dictionary for two tokens with identical atom lists
    dictionary = { 'ham': [[ 1, 2, 8 ]], 'spam': [[ 4, 6, 8 ]], 'eggs': [[ 4, 6, 1, 1 ]], 'pepper': [[ 4, 6 ]], 'bacon': [[ 1,2,8 ]] }

    # WITHOUT language model
    these_beams = trivial_beams.copy()
    these_beams['lm_weight'] = 1.0
    search_tree = SearchTree(phone_list,vocabulary,dictionary,these_beams,None)

    expected_current,expected_final = add_some_hypos(search_tree)

    check_tree_hypos(search_tree,expected_current,'current')
    check_tree_hypos(search_tree,expected_final,'final')

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

    search_tree = SearchTree(phone_list,vocabulary,dictionary,these_beams,lm)

    expected_current,expected_final = add_some_hypos(search_tree,lm)

    check_tree_hypos(search_tree,expected_current,'current')
    check_tree_hypos(search_tree,expected_final,'final')

def test_pruning_works():
    # build example tree
    phone_list = [ 1, 2, 4, 6, 8 ]
    vocabulary = [ 'ham', 'spam', 'eggs', 'pepper' ]
#     dictionary = { 'ham': [ 1, 2, 8 ], 'spam': [ 4, 6, 8 ], 'eggs': [ 4, 6, 1, 1 ], 'pepper': [ 4, 6 ] }

    search_tree = SearchTree(phone_list,vocabulary,Dictionary,{'hypo_beam': 1.2, 'final_beam': 3.4, 'hypo_topn': 3, 'final_topn': 2, 'token_insertion_penalty': 0, 'lm_weight': 0},None)
    search_tree.clear_hypos()

    # add final hypos
    ham_end_index,ham_end = trace_path(search_tree,[1,2,8])
    spam_end_index,spam_end = trace_path(search_tree,[4,6,8])

    search_tree.make_hypo(3.4,('spam','ham'),ham_end_index,'final')
    search_tree.make_hypo(1.9,('eggs','ham'),ham_end_index,'final')
    search_tree.make_hypo(5.9,('pepper','ham'),ham_end_index,'final')
    search_tree.make_hypo(4.9,('pepper','spam'),spam_end_index,'final')

    # add normal hypos
    ham_start_index,ham_start_node = trace_path(search_tree,[1])
    eggs_index,eggs_node = trace_path(search_tree,[4,6,1])
    branching_index,branching_node = trace_path(search_tree,[4,6])

    search_tree.make_hypo(2.22,('spam',),ham_start_index,'current')
    search_tree.make_hypo(1.11,('eggs',),ham_start_index,'current')
    search_tree.make_hypo(3.33,('ham',),ham_start_index,'current')
    search_tree.make_hypo(4.44,('ham','ham'),ham_start_index,'current')

    search_tree.make_hypo(5,('spam','ham','eggs'),eggs_index,'current')
    search_tree.make_hypo(6,('spam','eggs'),eggs_index,'current')
    search_tree.make_hypo(7,('ham','eggs'),eggs_index,'current')

    search_tree.make_hypo(8,('ham',),branching_index,'current')

    search_tree.prune_tree()

    expected_current = {
        ham_start_index: [(2.22,('spam',)),(1.11,('eggs',)),(3.33,('ham',))],
        eggs_index: [(5,('spam','ham','eggs')),(6,('spam','eggs')),(7,('ham','eggs'))],
        branching_index: [(8,('ham',))]
        }

    expected_final = {
        ham_end_index: [(3.4,('spam','ham')),(1.9,('eggs','ham'))],
        spam_end_index: [(4.9,('pepper','spam'))]
        }

    check_tree_hypos(search_tree,expected_current,'current')
    check_tree_hypos(search_tree,expected_final,'final')


def test_trace_nbest_works():
    # build example tree
    phone_list = [ 1, 2, 4, 6, 8 ]
    vocabulary = [ 'ham', 'spam', 'eggs', 'pepper' ]
#     dictionary = { 'ham': [ 1, 2, 8 ], 'spam': [ 4, 6, 8 ], 'eggs': [ 4, 6, 1, 1 ], 'pepper': [ 4, 6 ] }

    search_tree = SearchTree(phone_list,vocabulary,Dictionary,trivial_beams,None)
    search_tree.clear_hypos()

    # add some FINAL hypos
    ham_start_index,ham_start_node = trace_path(search_tree,[1])
    ham_end_index,ham_end = trace_path(search_tree,[1,2,8])
    spam_end_index,spam_end = trace_path(search_tree,[4,6,8])
    eggs_end_index,eggs_end = trace_path(search_tree,[4,6,1,0,1])
    pepper_end_index,pepper_end = trace_path(search_tree,[4,6])

    search_tree.make_hypo(3.4,('spam','ham'),ham_end_index,'final')
    search_tree.make_hypo(1.9,('eggs','ham'),ham_end_index,'final')
    search_tree.make_hypo(2.22,('spam',),spam_end_index,'final')
    search_tree.make_hypo(2.5,('eggs','eggs',),eggs_end_index,'final')
    search_tree.make_hypo(5,('spam','ham','eggs'),eggs_end_index,'final')

    # add some normal hypos (should be ignored)
    search_tree.make_hypo(0.1,('pepper','ham'),ham_start_index,'current')
    search_tree.make_hypo(3.4,('spam','ham'),ham_end_index,'current')
    search_tree.make_hypo(1.9,('eggs',),ham_end_index,'current')

    expected_nbest = [ (1.9,('eggs','ham')),(2.22,('spam',)),(2.5,('eggs','eggs',)),(3.4,('spam','ham')),(5,('spam','ham','eggs')) ]

    result = search_tree.trace_nbest()

    assert result == expected_nbest

def run_search_for_test(tree, neglog_probs):
    raise Exception('AAALT')
    tree.clear_hypos()
    print('Running search: time step 0, initializing')
    print('root node ids are',tree.root_nodes,' where length of "nodes" is',len(tree.nodes))
    print('root node atoms are:',[tree.nodes[i].atom_id for i in tree.root_nodes.values()])
    tree.initialize_search(neglog_probs[0,:])
#     pdb.set_trace()
    tree.finish_propagation()
    
    for time_step in range(1,neglog_probs.shape[0]):
#         pdb.set_trace()
#         outfile = open('DUMP_HYPOS.%02d.txt' % time_step,'w')
#         tree.dump_hypos(fid=outfile)
#         outfile.close()
        tree.propagate(neglog_probs[time_step,:])
        tree.propagate_external(neglog_probs[time_step,:])
        tree.finish_propagation()
        tree.prune_tree()

    pass

def test_entire_search():
    phone_list = list(range(0,10))
    vocabulary = [ 'ham', 'spam', 'eggs', 'sauce', 'mushrooms', 'spices', 'pepper' ]
    search_tree = SearchTree(phone_list,vocabulary,Dictionary,trivial_beams,None)

    data = pickle.load(open('search_test_data.two.pickle','r'))

#     run_search_for_test(search_tree,data['61148_simple'])
    search_tree.run_search(data['61148_simple'])
    result = search_tree.trace_nbest()
    minima = data['61148_simple'].argmin(axis=1)
    expected_best_score = sum([data['61148_simple'][time,minima[time]] for time in range(minima.shape[0])])

    assert np.isclose(expected_best_score,result[0][0])
    assert result[0][1][0] == 'spices'
    
    search_tree.run_search(data['61148_complex'])
    result = search_tree.trace_nbest()
    minima = data['61148_complex'].argmin(axis=1)
    expected_best_score = sum([data['61148_complex'][time,minima[time]] for time in range(minima.shape[0])])

    assert np.isclose(expected_best_score,result[0][0])
    assert result[0][1][0] == 'spices'

    search_tree.run_search(data['1284661'])
    result = search_tree.trace_nbest()
    minima = data['1284661'].argmin(axis=1)
    expected_best_score = sum([data['1284661'][time,minima[time]] for time in range(minima.shape[0])])

    assert np.isclose(expected_best_score,result[0][0])
    assert result[0][1] == ('ham','pepper','mushrooms')
