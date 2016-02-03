from __future__ import division, print_function

cimport numpy as np
import numpy as np
# cimport brainstorm._lm 
import operator
import time
import sys

from brainstorm._lm import LanguageModel # cimport does not help, really

# cdef struct Hypo:
#     cdef list history # tokens completely recognized so far, as strings
#     cdef double score


# cdef class Hypo:
#     # TOTAL score accumulated so far (includes LM history score)
#     cdef public double score
#     # tokens recognized so far, gets updated when a 'final' node is reached
#     # (including addition of LM score)
#     cdef public tuple history
# 
#     def __init__(self,double score,tuple history):
#         self.score = score
#         self.history = history
# 
#     def __str__(self):
#         return 'Hypo: score %f, history %s' % (self.score,self.history)

# A hopefully efficient container for hypos. Its main function is to assure
# that only ONE hypo is present for each history. Also allows pruning
# hypos according to their score.
# Iterating over this container yields (score, history) tuples
# TODO speed up!!
cdef class HypoContainer:
    # A dictionary indexed by history, containing exactly one score
    # per history
    cdef dict hypos

    # Constructor
    def __init__(self):
        self.hypos = {}

    # Insert a hypo into the container, assuring that of two hypos with the same
    # history, only the best one is kept
    cdef insert(self,double score,tuple history):
        cdef double old_score
        old_score = self.hypos.get(history,np.inf)
        self.hypos[history] = min(old_score,score)

    # delete all elements
    cdef clear(self):
        self.hypos = {}

    # Remove all hypos except the n best ones (with lowest score)
    cdef keep_nbest(self,int n):
        cdef list all_elements
        cdef tuple history
        if len(self.hypos) <= n:
            return
        all_elements = sorted(self.hypos.items(),key=operator.itemgetter(1))
        for history in (x[0] for x in all_elements[n:]):
            del self.hypos[history]

    # get all hypos as an iterator object (score, history)
    cdef public get_hypo_iterator(self):
        it = ( (x[1],x[0]) for x in self.hypos.items())        
        return it

    # move content to a different object
    cdef move_to(self,HypoContainer other_container):
        other_container.hypos = self.hypos
        self.hypos = {}

    # wrappers???
    def p_insert(self,double score,tuple history):
        return self.insert(score,history)

    def p_clear(self):
        return self.clear()

    def p_keep_nbest(self,n):
        return self.keep_nbest(n)

    def p_get_hypo_iterator(self):
        return self.get_hypo_iterator()

cdef class SearchNode:
    # atom attached to this node
    cdef public int atom_id
    # tokens (may be several ones if there are identically pronounced words)
    cdef public list tokens
    # current hypos (updated in finish_propagation)
    cdef public HypoContainer current_hypos
    # previous hypos (updated in finish_propagation)
    cdef public HypoContainer next_hypos
    # current final hypos (updated in finish_propagation)
    cdef public HypoContainer current_final_hypos
    # maps atom ids to child node INDICES in the nodes list
    cdef public dict child_nodes


    def __str__(self):
        return 'Atom: %d, child atoms -> indices: %s' % (self.atom_id,self.child_nodes)



# externally available main class (to be built before decoding starts)
cdef class SearchTree:
    cdef public list nodes
    cdef public dict root_nodes # list of INDICES
    cdef public set final_nodes # set of INDICES
    cdef public list phone_list
    cdef public list vocabulary
    cdef public dict dictionary
    cdef public dict beams
    cdef public object language_model

    # for pruning
    cdef public double next_max_score
    cdef public double next_max_final_score

#     cdef brainstorm._lm.LanguageModel language_model

    # Initializes this object. phone_list and vocabulary are list, dictionary is a 
    # Python dict (word => phone list), language_model (optional) is an instance of the
    # class LanguageModel.
    def __init__(self,phone_list,vocabulary,dictionary,beams,language_model):
        self.nodes = []
        self.root_nodes = {}
        self.final_nodes = set([])
        self.phone_list = phone_list
        self.vocabulary = vocabulary
        self.dictionary = dictionary
        self.beams = beams
        assert set(beams.keys()) == set(['hypo_beam','hypo_topn','final_beam','final_topn', 'token_insertion_penalty','lm_weight'])
        self.language_model = language_model

        self.make_tree_structure()

    ############# STATIC TREE STRUCTURE ############

    # make search node
    cdef SearchNode _make_node(self,atom_id):
        cdef SearchNode result
        result = SearchNode()
        result.atom_id = atom_id
        result.tokens = []
        result.current_hypos = HypoContainer()
        result.next_hypos = HypoContainer()
        result.current_final_hypos = HypoContainer()
        result.child_nodes = {}
        return result

    # Add a node to the tree. 
    cdef int _add_node(self,SearchNode new_node):
# # #         print('Add node')
        cdef int new_node_pos
# #         cdef SearchNode parent_node
        new_node_pos = len(self.nodes)
        self.nodes.append(new_node)
# #         if parent_node is not None:
# #             assert new_node.atom_id not in parent_node.child_nodes.keys()
# #             parent_node.child_nodes[new_node.atom_id] = new_node_pos
# #         else:
# #             assert new_node.atom_id not in self.root_nodes.keys()
# # #             print('Adding root node!')
# #             self.root_nodes[new_node.atom_id] = new_node_pos
        return new_node_pos
    
    # Look up node in dict, or create if necessary. Returns a tuple
    # (node_index, node, was_created) and inserts the node into parent_dict if necessary
    cdef tuple _make_or_create_node(self,dict parent_dict,int atom_id):
        try:
            node_index = parent_dict[atom_id]
            node = self.nodes[node_index]
            return (node_index,node,False)
        except KeyError:
            node = self._make_node(atom_id)
            node_index = self._add_node(node)
            parent_dict[atom_id] = node_index
            return (node_index,node,True)


    # Add a path to the tree, taking blank nodes into account.
    cdef void _add_path(self,list atom_ids,str final_token):
        # first find blank root node, or create it
        (blank_root_node_index,blank_root_node,_) = self._make_or_create_node(self.root_nodes,0)

        # find root node corresponding to first atom, or create it
        (atom_root_node_index,atom_root_node,was_created) = self._make_or_create_node(self.root_nodes,atom_ids[0])
        if was_created:
            blank_root_node.child_nodes[atom_ids[0]] = atom_root_node_index

        last_atom_node_index = atom_root_node_index
        last_atom_node = atom_root_node

        # iterate
        for atom in atom_ids[1:]:
            (blank_node_index,blank_node,_) = self._make_or_create_node(last_atom_node.child_nodes,0)
            (atom_node_index,atom_node,was_created) = self._make_or_create_node(blank_node.child_nodes,atom)
            if was_created and atom != last_atom_node.atom_id:
                # direct jump possible
                assert atom not in last_atom_node.child_nodes.keys()
                last_atom_node.child_nodes[atom] = atom_node_index

            last_atom_node_index = atom_node_index
            last_atom_node = atom_node

        # add last blank node, if it doesn't exist
        (blank_final_node_index,blank_final_node,_) = self._make_or_create_node(last_atom_node.child_nodes,0)

        # finished, now add tokens
        last_atom_node.tokens.append(final_token)
        self.final_nodes.add(last_atom_node_index)
        blank_final_node.tokens.append(final_token)
        self.final_nodes.add(blank_final_node_index)

    # create entire tree structure
    def make_tree_structure(self):
        cdef SearchNode new_node
        cdef str token
        cdef list atoms
        # iterate through vocabulary and call add_path appropiately
        for token in self.vocabulary:
            for atoms in self.dictionary[token]:
                self._add_path(atoms,token)

    ############# NODE DYNAMICS ############

    # Make a hypo and attach it to next_hypos. Note that the history is a tuple so that 
    # it is immutable
    # TODO happens only if history is not present in target node!!!
    cdef _make_hypo_to_next(self,double score,tuple history,int node_index):
        cdef HypoContainer next_hypos = self.nodes[node_index].next_hypos
        next_hypos.insert(score,history)
#         self.nodes[node_index].next_hypos.insert(score,history)

    # Make a hypo and attach it to current_final_hypos. Note that the history is a tuple so that 
    # it is immutable
    cdef _make_hypo_to_final(self,double score,tuple history,int node_index):
        cdef HypoContainer current_final_hypos = self.nodes[node_index].current_final_hypos
        current_final_hypos.insert(score,history)


    # Make a hypo (for test only), add it to the current, next, or current final hypos
    def make_hypo(self,double score,tuple history,int node_index,str what):
        if what == 'current':
            self.nodes[node_index].current_hypos.p_insert(score,history)
        elif what == 'next':
            self.nodes[node_index].next_hypos.p_insert(score,history)
        elif what == 'final':
            self.nodes[node_index].current_final_hypos.p_insert(score,history)
        else:
            raise Exception('make_hypo: parameter %s must be current, next, or final' % what)


    # Clear all hypos, so that a new search can be initialized
    def clear_hypos(self):
        cdef int node_index
        cdef SearchNode this_node
        for node_index in range(len(self.nodes)):
            this_node = self.nodes[node_index]
            this_node.current_hypos.clear()
            this_node.next_hypos.clear()
            this_node.current_final_hypos.clear()

        self.next_max_score = np.inf
        self.next_max_final_score = np.inf

    # Add initial hypos to the tree. Called to initialize a search, in lieu of propagate. After
    # this function, finish_propagation should be called.
    # neglog_probs is the  first frame of the CTC output prediction
    def initialize_search(self,np.ndarray[double, ndim=1] neglog_probs):
        cdef int root_node_atom
        cdef int root_node_index
        for (root_node_atom,root_node_index) in self.root_nodes.iteritems():
            self._make_hypo_to_next(neglog_probs[root_node_atom],(),root_node_index)

    # Recursion for propagate
    cdef tuple propagate_recursively(self,search_tree,start_node_index,np.ndarray[double, ndim=1] neglog_probs):
        cdef SearchNode this_node
        cdef int child_node_atom
        cdef int child_node_index

        cdef int propagated_hypos 
        cdef int deleted_hypos 
        cdef int sub_propagated_hypos 
        cdef int sub_deleted_hypos 
        propagated_hypos = 0
        deleted_hypos = 0

        this_node = search_tree.nodes[start_node_index]
        # propagate from here
        for (score,history) in this_node.current_hypos.get_hypo_iterator():
            if score <= search_tree.next_max_score:
                # self loop (CTC nothing)
                self._make_hypo_to_next(score + neglog_probs[0],history,start_node_index)
                # children
                for (child_node_atom,child_node_index) in this_node.child_nodes.items():
                    self._make_hypo_to_next(score + neglog_probs[child_node_atom],history,child_node_index)
                propagated_hypos += 1
            else:
                deleted_hypos += 1

        # recurse
        for child_node_index in this_node.child_nodes.values():
            (sub_propagated_hypos,sub_deleted_hypos) = self.propagate_recursively(search_tree,child_node_index,neglog_probs)
            propagated_hypos += sub_propagated_hypos
            deleted_hypos += sub_deleted_hypos

        return (propagated_hypos,deleted_hypos)

    # Propagate all hypos to child nodes. Negative log probs of the corresponding atoms are added. 
    # Note that the hypos are not yet made final (this happens only in the finish_propagation step).
    def propagate(self,np.ndarray[double, ndim=1] neglog_probs):

        cdef int node_index
        cdef SearchNode node
        cdef int propagated_hypos = 0
        cdef int deleted_hypos = 0
#         cdef int sub_propagated_hypos = 0
#         cdef int sub_deleted_hypos = 0

# #         print('Propagate: root nodes are',self.root_nodes)
#         for root_node_index in self.root_nodes.values():
#             (sub_propagated_hypos,sub_deleted_hypos) = self.propagate_recursively(self,root_node_index, neglog_probs)
#             propagated_hypos += sub_propagated_hypos
#             deleted_hypos += sub_deleted_hypos
        for node_index,node in enumerate(self.nodes):
            # propagate from here
            for (score,history) in node.current_hypos.get_hypo_iterator():
                if score <= self.next_max_score:
                    # self loop for this atom id, unchanged history
                    self._make_hypo_to_next(score + neglog_probs[node.atom_id],history,node_index)
                    # children
                    for (child_node_atom,child_node_index) in node.child_nodes.items():
                        self._make_hypo_to_next(score + neglog_probs[child_node_atom],history,child_node_index)
                    propagated_hypos += 1
                else:
                    deleted_hypos += 1

#         print('Propagate: forwarded %d hypos, deleted %d hypos' % (propagated_hypos,deleted_hypos))

    # Propagate all hypos marked as final (this marking must have happened in a previous call
    # to finish_propagation)
    def propagate_external(self,np.ndarray[double, ndim=1] neglog_probs):
        cdef int this_node_index
        cdef SearchNode this_node
        for this_node_index in self.final_nodes:
            this_node = self.nodes[this_node_index]
            for (score,history) in this_node.current_final_hypos.get_hypo_iterator():
                if score <= self.next_max_final_score:
                    for (root_node_atom,root_node_index) in self.root_nodes.items():
                        self._make_hypo_to_next(score + neglog_probs[root_node_atom],history,root_node_index)


    # copy 'next' to 'current' and 'final' hypos (in the latter case, the history gets extended, and a language model
    # score gets added
    def finish_propagation(self):
        cdef int this_node_index
        cdef SearchNode this_node
        for this_node_index in range(len(self.nodes)):
            this_node = self.nodes[this_node_index]
#             this_node.current_nodes = []
            this_node.current_final_hypos.clear()
            # token finalization
            for token in this_node.tokens:
                for (score,history) in this_node.next_hypos.get_hypo_iterator():
                    new_history = history + (token,)
                    # note that score = negative of computeProbability
                    lm_score = 0.0 if self.language_model is None else -self.language_model.computeProbability(new_history)
                    assert lm_score >= 0.0, 'Language model gave negative score %f to sequence %s' % (lm_score,str(new_history))
                    self._make_hypo_to_final(score + self.beams['lm_weight'] * lm_score + self.beams['token_insertion_penalty'],new_history,this_node_index)
            # otherwise just copy
            this_node.next_hypos.move_to(this_node.current_hypos) # clears out automatically

    # Prune tree (yet to see how). To be called after finish_propagation.
    def prune_tree(self):

        cdef SearchNode node

        # Step 2: find overall current best scores, set parameters for next step

        # score: small is beautiful
        cdef double lowest_score = np.inf
        cdef double lowest_final_score = np.inf
        for node in self.nodes:
            for (score,history) in node.current_hypos.get_hypo_iterator():
                if score < lowest_score:
                    lowest_score = score
            for (score,history) in node.current_final_hypos.get_hypo_iterator():
                if score < lowest_final_score:
                    lowest_final_score = score

        self.next_max_score = lowest_score + self.beams['hypo_beam']
        self.next_max_final_score = lowest_score + self.beams['final_beam']
#         print('Setting new pruning scores:',self.next_max_score,self.next_max_final_score)

        # Step 3: prune hypos according to topN
        for node in self.nodes:
            node.current_hypos.keep_nbest(self.beams['hypo_topn'])
            node.current_final_hypos.keep_nbest(self.beams['final_topn'])
#             if len(node.current_hypos) > self.beams['hypo_topn']:
#                 node.current_hypos.sort(key=lambda h: h.score)
#                 node.current_hypos[self.beams['hypo_topn']+1:] = []
#             if len(node.current_final_hypos) > self.beams['final_topn']:
#                 node.current_final_hypos.sort(key=lambda h: h.score)
#                 node.current_final_hypos[self.beams['final_topn']+1:] = []

         
    # TEST ONLY
    def dump_hypo_container(self,HypoContainer hc,nbest,fid):
        all_hypos = hc.hypos.items()
        all_hypos.sort(key=operator.itemgetter(1))
        out = str(all_hypos[0:nbest])
        print(out,file=fid)

    def dump_hypos(self,from_node_index = -1,nbest = 10,fid=sys.stdout):
        all_nodes = [ from_node_index ] if from_node_index >= 0 else list(range(len(self.nodes)))

        for node_index in all_nodes:
            print('Node %d with atom %d, tokens %s' % (node_index,self.nodes[node_index].atom_id,str(self.nodes[node_index].tokens)),file=fid)
            print('CURRENT Hypos:',end='',file=fid)
            self.dump_hypo_container(self.nodes[node_index].current_hypos,nbest,fid)
            print('NEXT Hypos:',end='',file=fid)
            self.dump_hypo_container(self.nodes[node_index].next_hypos,nbest,fid)
            print('FINAL Hypos:',end='',file=fid)
            self.dump_hypo_container(self.nodes[node_index].current_final_hypos,nbest,fid)


    # Run an entire search. neglog_probs must be two-dimensional (time x prediction), note that each row
    # is one state of the CTC output layer, with position 0 referring to the special CTC 'blank' node
    def run_search(self,np.ndarray[double, ndim=2] neglog_probs):
        cdef int time_step

        self.clear_hypos()
#         print('Running search: time step 0, initializing')
#         print('root node ids are',self.root_nodes,' where length of "nodes" is',len(self.nodes))
#         print('root node atoms are:',[self.nodes[i].atom_id for i in self.root_nodes.values()])
        self.initialize_search(neglog_probs[0,:])
        self.finish_propagation()
        
        for time_step in range(1,neglog_probs.shape[0]):
            start_time = time.time()
            self.propagate(neglog_probs[time_step,:])
            self.propagate_external(neglog_probs[time_step,:])
            prop_time = time.time()
            self.finish_propagation()
            self.prune_tree()
            end_time = time.time()
#             print('Finished search: time step %d, prop took %f, finishing took %f' % (time_step,prop_time - start_time,end_time - prop_time))


    # apply to a tree after run_search has run through
    def trace_nbest(self,int n = 10):
        cdef int node_index
        cdef list hypo_list = []
        cdef HypoContainer cfh
        for node_index in self.final_nodes:
            cfh = self.nodes[node_index].current_final_hypos
            hypo_list.extend(cfh.get_hypo_iterator())
#             hypo_list.extend(self.nodes[node_index].current_final_hypos.get_hypo_iterator())

        hypo_list.sort(key=operator.itemgetter(0))
        return hypo_list[0:n]
        
        # I think we want to return a standard Python object
#         return [ (h.score,h.history) for h in hypo_list[0:n] ]




# # jump-in function
# def RunTreeSearch(ctc_prediction,search_tree):
#     pass
