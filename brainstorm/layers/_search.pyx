from __future__ import division, print_function

cimport numpy as np
# cimport brainstorm._lm 

from brainstorm._lm import LanguageModel # cimport does not help, really

# cdef struct Hypo:
#     cdef list history # tokens completely recognized so far, as strings
#     cdef double score

cdef class Hypo:
    # TOTAL score accumulated so far (includes LM history score)
    cdef public double score
    # tokens recognized so far, gets updated when a 'final' node is reached
    # (including addition of LM score)
    cdef public tuple history

    def __init__(self,double score,tuple history):
        self.score = score
        self.history = history

    def __str__(self):
        return 'Hypo: score %f, history %s' % (self.score,self.history)

# TODO could be struct??
cdef class SearchNode:
    # atom attached to this node
    cdef public int atom_id
    # tokens (may be several ones if there are identically pronounced words)
    cdef public list tokens
    # current hypos (updated in finish_propagation)
    cdef public list current_hypos
    # previous hypos (updated in finish_propagation)
    cdef public list next_hypos
    # current final hypos (updated in finish_propagation)
    cdef public list current_final_hypos
    # maps atom ids to child node INDICES in the nodes list
    cdef public dict child_nodes

    def __str__(self):
        return 'Atom: %d, child atoms -> indices: %s' % (self.atom_id,self.child_nodes)

# externally available main class (to be built before decoding starts)
cdef class SearchTree:
    cdef public list nodes
    cdef public dict root_nodes # list of INDICES
    cdef public list final_nodes # list of INDICES
    cdef public list phone_list
    cdef public list vocabulary
    cdef public dict dictionary
    cdef public object language_model
#     cdef brainstorm._lm.LanguageModel language_model

    # Initializes this object. phone_list and vocabulary are list, dictionary is a 
    # Python dict (word => phone list), language_model (optional) is an instance of the
    # class LanguageModel.
    def __init__(self,phone_list,vocabulary,dictionary,language_model):
        self.nodes = []
        self.root_nodes = {}
        self.final_nodes = []
        self.phone_list = phone_list
        self.vocabulary = vocabulary
        self.dictionary = dictionary
        self.language_model = language_model

        self.make_tree_structure()

    ############# STATIC TREE STRUCTURE ############

    # make search node
    cdef SearchNode _make_node(self,atom_id):
        cdef SearchNode result
        result = SearchNode()
        result.atom_id = atom_id
        result.tokens = []
        result.current_hypos = []
        result.next_hypos = []
        result.current_final_hypos = []
        result.child_nodes = {}
        return result

    # Add a node to the tree. If parent is None, this node is automatically added
    # to the list of root nodes. Returns the position of the new node
    cdef int _add_node(self,SearchNode new_node,SearchNode parent_node):
#         print('Add node')
        cdef int new_node_pos
        new_node_pos = len(self.nodes)
        self.nodes.append(new_node)
        if parent_node is not None:
            assert new_node.atom_id not in parent_node.child_nodes.keys()
            parent_node.child_nodes[new_node.atom_id] = new_node_pos
        else:
            assert new_node.atom_id not in self.root_nodes.keys()
#             print('Adding root node!')
            self.root_nodes[new_node.atom_id] = new_node_pos
        return new_node_pos

    # Add a path to the tree. Called recursively. When a final node is created
    # (by adding its first token), the node is added to final_tokens
    cdef void _add_path(self,int root_node_pos,list atom_ids,list final_tokens):
        cdef SearchNode root_node
        cdef SearchNode node
        cdef int node_index
        root_node = self.nodes[root_node_pos]
        if len(atom_ids) == 0:
            root_node.tokens += final_tokens
#             print('Append to final nodes')
            self.final_nodes.append(root_node_pos)
            return
        else:
            # take first atom, check whether the corresponding node exists, create
            # it if necessary, recurse

            try:
                node_index = root_node.child_nodes[atom_ids[0]]
            except KeyError:
                new_node = self._make_node(atom_ids[0])
                node_index = self._add_node(new_node,root_node)

            self._add_path(node_index,atom_ids[1:],final_tokens)
            return

    # create entire tree structure
    def make_tree_structure(self):
        cdef SearchNode new_node
        cdef str token
        # iterate through vocabulary and call add_path appropiately
        for token in self.vocabulary:
            atoms = self.dictionary[token]
           
            # find root node
            try:
                node_index = self.root_nodes[atoms[0]]
            except KeyError:
                new_node = self._make_node(atoms[0])
                node_index = self._add_node(new_node,None)
            self._add_path(node_index,atoms[1:],[token])

    ############# NODE DYNAMICS ############

    # Make a hypo and attach it to next_hypos. Note that the history is a tuple so that 
    # it is immutable
    cdef _make_hypo_to_next(self,double score,tuple history,int node_index):
        cdef Hypo hypo
        hypo = Hypo(score,history)
        self.nodes[node_index].next_hypos.append(hypo)
        return hypo

    # Make a hypo and attach it to current_final_hypos. Note that the history is a tuple so that 
    # it is immutable
    cdef _make_hypo_to_final(self,double score,tuple history,int node_index):
        cdef Hypo hypo
        hypo = Hypo(score,history)
        self.nodes[node_index].current_final_hypos.append(hypo)
        return hypo


    # Make a hypo (for test only), add it to the current, next, or current final hypos
    def make_hypo(self,double score,tuple history,int node_index,str what):
        cdef Hypo hypo
        hypo = Hypo(score,history)
        if what == 'current':
            self.nodes[node_index].current_hypos.append(hypo)
        elif what == 'next':
            self.nodes[node_index].next_hypos.append(hypo)
        elif what == 'final':
            self.nodes[node_index].current_final_hypos.append(hypo)
        else:
            raise Exception('make_hypo: parameter %s must be current or next' % what)

        return hypo
    # Propagate all hypos to child nodes. Negative log probs of the corresponding atoms are added. 
    # Note that the hypos are not yet made final (this happens only in the finish_propagation step)
    def propagate(self,np.ndarray[double, ndim=1] neglog_probs):
        def propagate_recursively(search_tree,start_node_index,np.ndarray[double, ndim=1] neglog_probs):
#             print('Enter RecProp',start_node_index)
            cdef SearchNode this_node
            cdef Hypo hypo
            cdef int child_node_atom
            cdef int child_node_index
            this_node = search_tree.nodes[start_node_index]
#             print('RecProp %d current hypos' % len(this_node.current_hypos))
            # propagate from here
            for hypo in this_node.current_hypos:
                # self loop (CTC nothing)
                self._make_hypo_to_next(hypo.score + neglog_probs[0],hypo.history,start_node_index)
                # children
                for (child_node_atom,child_node_index) in this_node.child_nodes.items():
                    self._make_hypo_to_next(hypo.score + neglog_probs[child_node_atom],hypo.history,child_node_index)

            # recurse
            for child_node_index in this_node.child_nodes.values():
                propagate_recursively(search_tree,child_node_index,neglog_probs)

        cdef int root_node_index

#         print('Propagate: root nodes are',self.root_nodes)
        for root_node_index in self.root_nodes.values():
           propagate_recursively(self,root_node_index, neglog_probs)

    # Propagate all hypos marked as final (this marking must have happened in a previous call
    # to finish_propagation)
    def propagate_external(self,np.ndarray[double, ndim=1] neglog_probs):
        for this_node_index in self.final_nodes:
            this_node = self.nodes[this_node_index]
            for hypo in this_node.current_final_hypos:
                for (root_node_atom,root_node_index) in self.root_nodes.items():
                    self._make_hypo_to_next(hypo.score + neglog_probs[root_node_atom],hypo.history,root_node_index)


    # copy 'next' to 'current' and 'final' hypos (in the latter case, the history gets extended, and a language model
    # score gets added
    def finish_propagation(self):
        for this_node_index in range(len(self.nodes)):
            this_node = self.nodes[this_node_index]
#             this_node.current_nodes = []
            this_node.current_final_hypos = []
            # token finalization
            for token in this_node.tokens:
                for hypo in this_node.next_hypos:
                    new_history = hypo.history + (token,)
                    lm_score = 0.0 if self.language_model is None else self.language_model.computeProbability(new_history)
                    self._make_hypo_to_final(hypo.score + lm_score,new_history,this_node_index)
            # otherwise just copy
            this_node.current_hypos = this_node.next_hypos
            this_node.next_hypos = [] # make sure there's just one link






# jump-in function
def RunTreeSearch(ctc_prediction,search_tree):
    pass
