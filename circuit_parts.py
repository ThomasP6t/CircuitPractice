import random
import logging
import elements
import numpy as np
from random import randint
from PIL import Image, ImageDraw
from itertools import product
from utils import get_divisors, partition, pprint_value, weighted_sample, \
    make_col_image, make_hop_image, standard_width, info_font, solution_color


class Node:
    count = 0
    def __init__(self, voltage=None, circuit=None):
        self.arrivals = []
        self.departures = []
        self.voltage = voltage
        self.circuit = circuit
        self.number = self.count
        self.__class__.count += 1

    def __repr__(self):
        return f"n{self.number}" + (f": {pprint_value(self.voltage)}V" if self.voltage is not None else '')

    def netlist_repr(self):
        return str(self.number)

    @property
    def solution_text(self):
        v = '?' if self.voltage is None or np.isnan(float(self.voltage)) else pprint_value(self.voltage)
        return f"V={v}V"

    def solution_bbox(self):
        """
        :return: tuple with width and height of element text
        """
        draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        bbox = draw.textbbox((0, 0), self.solution_text, font=info_font)
        return bbox[2], bbox[3]


class Metagraph:
    """
    Metagraph of a circuit:
        This contains only graphs that are a serial configuration of parallel arrangements of parts between two nodes.
        The parts are either Elements, or another (sub)graph with the same restrictions.
        The 'top' graph is between self.start and self.end
        All graphs generated by Circuit._build_topology can be broken down like this
    This allows for a view of the circuit where voltage and current assignment can be done recursively,
        while keeping 'round' numbers across the board
    Internal representation is in attributes path and :
        - path: a list of nodes that form the (series) path between the beginning and end nodes of the graph
        - hop_configs: is a list of the configurations through the path, at location i for the 'hop' between
              path[i] and path[i+1].
              each configuration (element of hop_configs) is a list of the parts that are all between the two
              nodes of that hop on the path (in a parallel fashion)
              a part is either an object of (a subclass of) elements.Element
                  or another instance of Metagraph, recursively with the same construction as described here.
                  It's turtles all the way down.
    """

    def __init__(self, circuit, no_compute=False):
        self.circuit = circuit
        if not no_compute:
            self._compute()
        self.hops_accepted_proposals = {}
        self.max_v_for_floating_part = 1

    def _compute(self, force=False):
        if not (hasattr(self, "path") and hasattr(self, "hop_configs")) or force:
            circuit = self.circuit
            layer_layout = self.circuit._layer_layout()
            # make the graph of node connectivity
            graph = {}
            for n in self.circuit.nodes:
                if n != self.circuit.end:
                    graph[n] = {}
                    for el in n.departures:
                        n_to = el.conn_to
                        if n_to not in graph[n]:
                            graph[n][n_to] = []
                        graph[n][n_to].append(el)
            # find the lowest split
            split_nodes = [n for n in graph.keys() if len(graph[n]) > 1]
            while split_nodes:
                last_split = max(split_nodes, key=lambda n: layer_layout[n])
                # ...and from there, the highest rejoin
                next_nodes = list(graph[last_split].keys())
                following_nodes = next_nodes
                while len(set(following_nodes)) == len(set(next_nodes)):
                    next_node_layers = [layer_layout[n] for n in following_nodes]  # look at layer heights...
                    next_following_nodes = []
                    for n in following_nodes:
                        if layer_layout[n] == min(next_node_layers):
                            # graph[n] is a set with always 1 element (no splits after last_split))
                            next_following_nodes.append(list(graph[n].keys())[0])
                        else:  # if not on highest layer, keep that current node
                            next_following_nodes.append(n)
                    following_nodes = next_following_nodes
                # since we started from the last split, there are certainly not more elements in set(following_nodes),
                # but less => find the elements that are equal
                following_nodes = np.array(following_nodes)  # pull into numpy, so we can do vector logic
                num_occurences = [sum(following_nodes == n) for n in following_nodes]  # frequency count
                rejoin_node = following_nodes[np.argmax(num_occurences)]  # find one of the nodes that maximally rejoins
                rejoin_locs = np.flatnonzero(
                    following_nodes == rejoin_node)  # all operations have kept order of next_nodes
                # now replace all circuitry between last_split and rejoin_node by a single 'edge' = another graph
                # do this by 'column' first = per rejoin_loc
                # first create the part that will become graph[first_split][rejoin_node]
                new_part = []
                for rejoin_loc in rejoin_locs:
                    n_from, n_to = last_split, next_nodes[rejoin_loc]
                    if n_to == rejoin_node:  # direct connection last_split - rejoin_node, so no need to make dict
                        new_part.extend(graph[n_from][n_to])
                        continue
                    col_parts = {}  # our subgraph
                    col_parts[n_from] = {n_to: graph[n_from][n_to]}
                    while n_to != rejoin_node:
                        n_from, n_to = n_to, list(graph[n_to].keys())[0]
                        col_parts[n_from] = {n_to: graph[n_from][n_to]}
                    if rejoin_node not in graph[last_split]:
                        graph[last_split][rejoin_node] = []
                    new_part.append(col_parts)
                # now do the iteration again to delete current parts
                for rejoin_loc in rejoin_locs:
                    n_from, n_to = last_split, next_nodes[rejoin_loc]
                    while True:
                        del graph[n_from][n_to]
                        if not graph[n_from]:  # empty dict
                            del graph[n_from]
                        if n_to == rejoin_node:
                            break
                        n_from, n_to = n_to, list(graph[n_to].keys())[0]
                # finally add to graph
                if last_split not in graph:
                    graph[last_split] = {}
                graph[last_split][rejoin_node] = new_part
                # and start over with the new graph
                split_nodes = [n for n in graph.keys() if len(graph[n]) > 1]
            # now split_nodes is empty, so graph is one column

            # convert to Metagraph of Metagraphs
            def convert(subgraph, start_node=self.circuit.start, top_level=False):  # conversion to final format
                if not isinstance(subgraph, dict):  # Element
                    return subgraph
                path, hop_configs = [start_node], []
                # because of the method above, subgraph must be all series: each value of subgraph is a 1-element dict
                node_from, node_to = start_node, list(subgraph[start_node].keys())[0]
                while True:
                    path.append(node_to)
                    hop_configs.append([convert(subsubgraph, start_node=node_from)
                                        for subsubgraph in subgraph[node_from][node_to]])
                    if node_to not in subgraph:  # reached the end
                        break
                    node_from, node_to = node_to, list(subgraph[node_to].keys())[0]
                if top_level:  # no need to make a new object
                    return path, hop_configs
                result = Metagraph(circuit, no_compute=True)
                result.path, result.hop_configs = path, hop_configs
                return result

            self.path, self.hop_configs = convert(graph, top_level=True)

    def __repr__(self, indent=1):
        tab = '\t'
        result = "Metagraph object"
        for i, hop_config in enumerate(self.hop_configs):
            result += f"\n{tab*indent}hop {i+1}: {self.path[i]} -> {self.path[i+1]}, parts\n"
            result += '\n'.join([f"{tab*(indent+1)}- {part.__repr__(indent=indent+2)}" for part in hop_config])
        return result

    @property
    def hops_can_be_undetermined(self):
        """ returns for each hop if there is room for an undetermined part"""
        hop_undetermined = []
        for hop_config in self.hop_configs:
            hop_undetermined.append(False)
            for part in hop_config:
                if part.can_be_undetermined:
                    hop_undetermined[-1] = True
                    break
        return hop_undetermined

    @property
    def can_be_undetermined(self):
        return all(self.hops_can_be_undetermined)

    @property
    def hops_can_be_zero(self):
        """ returns for each hop whether it can have a current of 0 while still having a voltage over it"""
        return [all([part.can_be_zero for part in hop_config]) for hop_config in self.hop_configs]

    @property
    def can_be_zero(self):
        """ returns whether the metagraph can have a current of 0 while still having a voltage over it"""
        return any(self.hops_can_be_zero)

    @property
    def undetermined(self):
        for hop_config in self.hop_configs:
            if not any([part.undetermined for part in hop_config]):  # one hop with all parts determined is enough
                return False
        return True

    @property
    def open_for_i_change(self):
        # one hop's openness ~= max of part opennesses
        # for full series, take product (corollary: if one hop has 0 openness, then full path also)
        return np.prod([max([part.open_for_i_change for part in hop_config]) for hop_config in self.hop_configs])

    def i_acceptable(self, i, v=None):
        if any([np.isnan(hop_v) for hop_v in self.hops_v]):  # special case: if part of the hops are floating
            return i == 0  # only 0 current is ok
        if v is None:
            v_partitions = [[None] * len(self.hop_configs)]
        elif v == 0 or np.isnan(float(v)):
            if i == 0:  # (0, 0) and (0, nan) are always acceptable
                self.hops_accepted_proposals[(0, v)] = [[0 for _ in hop_config] for hop_config in self.hop_configs], \
                                                       [v for _ in self.hop_configs]
                for hop_config in self.hop_configs:  # propagate to put in part.hops_accepted_proposals
                    for part in hop_config:
                        part.i_acceptable(0, v)
            return i == 0
        else:
            if i > 0:  # hops_can_be_zero is only relevant for non-zero current
                v_partitions = partition(self.hops_can_be_zero, v)
            else:
                v_partitions = partition([True] * len(self.hop_configs), v)
        for v_partition in v_partitions:
            hops_accepted_proposal = []
            for hop_config, hop_v in zip(self.hop_configs, v_partition):
                part_proposals = [part.proposed_i for part in hop_config]
                # the sum of part currents needs to be set to i
                proposal_delta = i - sum(part_proposals)
                opennesses = [part.open_for_i_change for part in hop_config]
                if proposal_delta == 0:  # no change so no openness needed
                    opennesses = [max(o, 0.01) for o in opennesses]
                # for now, try all non-0, in descending order of openness
                index_order = sorted(np.flatnonzero(opennesses), key=lambda i: opennesses[i], reverse=True)
                accepted_proposal = False
                # 2 runs: first try to get the full change in one part, then try a partial fill
                for index in index_order:
                    full_change = part_proposals.copy()
                    old_proposal = part_proposals[index]
                    if old_proposal + proposal_delta < 0:  # i is a decrease of more than old_proposal
                        continue
                    if hop_config[index].i_acceptable(old_proposal + proposal_delta, hop_v):
                        accepted_proposal = True
                        for other_index in range(len(hop_config)):  # still need to check if voltage ok for other parts
                            if other_index != index:
                                if not hop_config[other_index].i_acceptable(part_proposals[other_index], hop_v):
                                    accepted_proposal = False
                                    break
                        if accepted_proposal:
                            full_change[index] = old_proposal + proposal_delta
                            hops_accepted_proposal.append(full_change)
                            break
                if not accepted_proposal and len(index_order) > 1:
                    # 2nd run: propose all changes up to proposal_delta, and try the rest in following parts
                    # for efficiency don't try all combinations, but continue with the biggest change accepted
                    partial_changes = part_proposals.copy()
                    partial_changes_locs = set()
                    for index in index_order:
                        old_proposal = part_proposals[index]
                        if proposal_delta > 0:
                            tries = range(proposal_delta, 0, -1)
                        else:  # proposal_delta < 0
                            tries = range(max(proposal_delta, -old_proposal), 0)
                        for t in tries:
                            if hop_config[index].i_acceptable(old_proposal + t, hop_v):
                                partial_changes[index] = old_proposal + t
                                proposal_delta -= t
                                partial_changes_locs.add(index)
                                if proposal_delta == 0:
                                    accepted_proposal = True
                                    hops_accepted_proposal.append(partial_changes)
                                break  # stop trying for this part
                        if accepted_proposal:
                            for other_index in range(len(hop_config)):  # check if voltage ok for other parts
                                if other_index not in partial_changes_locs:
                                    if not hop_config[other_index].i_acceptable(part_proposals[other_index], hop_v):
                                        accepted_proposal = False
                                        break
                            break
                if not accepted_proposal:  # if one hop fails, the whole chain fails
                    break
            if accepted_proposal:
                self.hops_accepted_proposals[(i, v)] = hops_accepted_proposal, list(v_partition)  # save for reuse
                return True
        return False

    def change_i_v_proposal(self, i, v=None):
        accepted = False
        if (i, v) in self.hops_accepted_proposals:
            accepted = True
        elif np.isnan(float(v)):  # float('nan') != np.nan ...
            nan_i = [key_i for key_i, key_v in self.hops_accepted_proposals.keys() if np.isnan(float(key_v))]  # NANI ??
            if i in nan_i:
                v = [key_v for key_i, key_v in self.hops_accepted_proposals.keys()
                     if np.isnan(float(key_v)) and key_i == i][0]
                accepted = True
        elif any([np.isnan(hop_v) for hop_v in self.hops_v]) and i == 0:  # special case: part of the hops are floating
            self.proposed_i = 0
            return  # no need to change anything else
        elif i == self.proposed_i and (v is None or v == sum(self.hops_v)):  # no changes needed
            return
        assert accepted, f"Something has gone wrong; {i} is not an acceptable current for subgraph {self}"
        self.proposed_i = i
        new_i_proposal, new_v_proposal = self.hops_accepted_proposals[(i, v)]
        self.hops_v = new_v_proposal
        new_hops_i_parts = []
        for hop_config, hop_i, hop_v in zip(self.hop_configs, new_i_proposal, new_v_proposal):
            new_hops_i_parts.append(hop_i.copy())
            for part, part_i in zip(hop_config, hop_i):
                part.change_i_v_proposal(part_i, v=hop_v)
        self.hops_i_parts = new_hops_i_parts


    def get_i_v(self, allow_undetermined=False, oom_center=None):
        """
        First trip down in the process of assigning voltages and currents.
            voltage: get minimum number of 'voltage parts' (as defined by CircuitState.voltage_step) needed
            current: assign the parts that will have the ability to be undetermined
                     and get a number of 'current parts' from Element.get_nice_current_parts
                         that satisfies the condition that these remain 'nice' independent of voltage
        :param allow_undetermined: if True, all hops can have one part undetermined
                                   if False, at least one hop needs all parts determined
                                   See below for the definition of determined
               oom_center: center to keep current proposal within 1-2 orders of magnitude around this value
        :return: None, all calculations are stored in self and sub-Metagraphs

        We introduce the concept of 'determined':
            - an element is determined if there exist a voltage and parameters that fix the given i
              (ex. resistor always, diode for i = 0)
            - a subgraph is determined if at least one of the serial hops has all parallel parts determined
                  and if for the other hops at most one of the parallel parts is undetermined
                  if that undetermined part is a subgraph, then all hops can have at most one part undetermined
            For a meta-graph that is determined, all currents are well-defined.
        """
        hops_allow_undetermined = [True] * len(self.hop_configs)
        hops_can_be_undetermined = self.hops_can_be_undetermined
        hops_min_v = [None] * len(self.hop_configs)
        hops_i_parts = [None] * len(self.hop_configs)
        hops_undetermined_parts = [None] * len(self.hop_configs)
        running_lcm = 1
        if not allow_undetermined:  # one hop must be determined
            if all(hops_can_be_undetermined):  # only when all can be undetermined, there is a need to intervene
                hops_allow_undetermined[randint(0, len(self.hop_configs)-1)] = False
        # to fully make random, first shuffle the hops
        shuffled_hop_indices = random.sample(range(len(self.hop_configs)), k=len(self.hop_configs))
        for hop_index in shuffled_hop_indices:
            hop_config = self.hop_configs[hop_index]
            hop_allow_undetermined = hops_allow_undetermined[hop_index]
            parts_allow_undetermined = [False] * len(hop_config)
            if hop_allow_undetermined:  # then one part can be undetermined
                parts_can_be_undetermined = [part.can_be_undetermined for part in hop_config]
                can_be_undetermined_locs = np.flatnonzero(parts_can_be_undetermined)
                if len(can_be_undetermined_locs) > 0:  # among those parts that can, pick one
                    undetermined_loc = random.choice(can_be_undetermined_locs)
                    parts_allow_undetermined[undetermined_loc] = True
            this_min_v_parts = [None] * len(hop_config)
            this_i_parts = [None] * len(hop_config)
            this_undetermined_parts = [None] * len(hop_config)
            # again random order
            shuffled_part_indices = random.sample(range(len(hop_config)), k=len(hop_config))
            for part_index in shuffled_part_indices:
                part = hop_config[part_index]
                part_allow_undetermined = parts_allow_undetermined[part_index]
                undetermined_part = False
                if isinstance(part, elements.Element):
                    if part_allow_undetermined:
                        if random.random() < self.circuit.circuitstate.undetermined_prob:  # throw the dice
                            undetermined_part = True
                    min_v = randint(1, 3)  # claim between 1 and 3 v-parts per element
                    i = part.get_nice_current_parts(make_undetermined=undetermined_part, oom_center=oom_center)
                else:  # part is a Metagraph
                    part.get_i_v(allow_undetermined=part_allow_undetermined, oom_center=oom_center)
                    min_v = sum(part.hops_v)
                    i = part.proposed_i
                    if any(np.isnan(part.hops_v)) and logging.getLogger().isEnabledFor(logging.DEBUG):
                        assert i == 0, f"Problem with floating voltages in part {part}"
                    # undetermined if each hop has undetermined part
                    undetermined_part = all([any(undetermined_parts)
                                             for undetermined_parts in part.hops_undetermined_parts])
                if oom_center is None and i > 0:
                    oom_center = i
                this_min_v_parts[part_index] = min_v
                this_i_parts[part_index] = i
                this_undetermined_parts[part_index] = undetermined_part

            if sum(this_i_parts) > 0 and running_lcm % sum(this_i_parts) != 0:
                # (try) to do some arbitrage to nudge the total current for this hop towards having a big gcd with
                # previous hops' current proposals
                # this is motivated by the theorem lcm(a,b) = a * b / gcd(a,b), and we want a low lcm
                opennesses = [part.open_for_i_change for part in hop_config]
                # for now, try all non-0, in descending order of openness
                arbitrage_success = False
                for part_index in sorted(np.flatnonzero(opennesses), key=lambda i: opennesses[i], reverse=True):
                    total_i_not_this_part = sum(this_i_parts) - this_i_parts[part_index]
                    # candidates for total current:
                    #     - not less than total_i_not_this_part + 1 (otherwise current for this part would be < 0
                    #     - not more than the new lcm (otherwise we'd be making matters worse)
                    if total_i_not_this_part <= 1:
                        continue
                    candidates = range(total_i_not_this_part+1, np.lcm(running_lcm, sum(this_i_parts)))
                    if running_lcm == 1:  # first hop -> favour low numbers with high # divisors
                        scores = [len(get_divisors(c))**.99/c for c in candidates]  # exponent < 1 => score(2x) < score(x)
                    else:
                        scores = np.gcd(candidates, running_lcm)
                    # to try and make a smart selection, we take a top (equal to amount with above-average score)
                    # then take a weighted sample of #top elements, weighted by score
                    top = sum(np.array(sorted(scores, reverse=True)) >= np.average(scores))
                    top = min(max(top, 3), len(candidates))  # at least go for a top 3, but at most the # candidates
                    for top_candidate in weighted_sample(candidates, scores, top):
                        proposal = top_candidate - total_i_not_this_part
                        accept = hop_config[part_index].i_acceptable(proposal)
                        if accept:
                            arbitrage_success = True
                            this_i_parts[part_index] = proposal
                            break
                    if arbitrage_success:
                        break

            this_min_v = np.nanmax(this_min_v_parts)
            if not np.isnan(this_min_v):
                this_min_v = int(this_min_v)
            hops_min_v[hop_index] = this_min_v  # keep one number per part
            for part, i_part in zip(self.hop_configs[hop_index], this_i_parts):  # update in each part
                # normally this should not be a problem...
                assert part.i_acceptable(i_part, v=this_min_v), \
                    "Problem in equalizing voltages across a parallel part. Part " + repr(part)
                part.change_i_v_proposal(i_part, this_min_v)
            hops_i_parts[hop_index] = this_i_parts  # keep all
            hops_undetermined_parts[hop_index] = this_undetermined_parts
            running_lcm = np.lcm(running_lcm, sum(this_i_parts))

        # current through all hops needs to be the same => equalize to lcm(i_parts)
        # this entails that for each hop, both i and min_v need to be multiplied by lcm / i (to keep 'niceness')
        # note that this assumes that such an operation is possible for all parts, this might not be the case for
        #     other Elements that get added later
        if logging.getLogger().isEnabledFor(logging.DEBUG) and running_lcm > 0:  # extra debug check
                assert np.lcm.reduce([sum(hop_i_parts) for hop_i_parts in hops_i_parts]) == running_lcm
        self.proposed_i = running_lcm
        if running_lcm > 0:
            # further arbitrage attempts: find ways to avoid setting i as high as lcm, and multiplying all v's
            #                             with lcm / sum(i's through hop)
            hops_v = []
            # TODO try also variations in i? certainly if v_multiplier small
            # improved_i = []
            # i_divs = np.array(get_divisors(running_lcm))
            # i_all_divs = set([i_divs[list(p)].prod() for p in product([True, False], repeat=len(i_divs))])
            for hop_index, hop_config in enumerate(self.hop_configs):
                v_multiplier = running_lcm // sum(hops_i_parts[hop_index])
                hop_min_v = hops_min_v[hop_index]
                this_i_parts = hops_i_parts[hop_index]
                if v_multiplier > 1:
                    v_multiplier_divs = np.array(get_divisors(v_multiplier))
                    v_all_divs = set([v_multiplier_divs[list(p)].prod() for p in product([True, False],
                                                                                         repeat=len(v_multiplier_divs))])
                    improved_v = []  # try to find an acceptable v that is less drastic than the lcm
                    for v in range(hop_min_v, hop_min_v*2):
                        for d in sorted(v_all_divs):
                            accept_d = True
                            for part, part_i in zip(hop_config, this_i_parts):
                                if not part.i_acceptable(v_multiplier * part_i, v=d*v):
                                    accept_d = False  # if one part doesn't accept, it's over
                                    break
                            if accept_d and d*v < hop_min_v * v_multiplier:
                                improved_v.append(d*v)
                    if improved_v:
                        # pick one, with better probability for lower solutions
                        improved_v = list(set(improved_v))
                        v = random.choices(improved_v, weights=1/np.array(improved_v), k=1)[0]
                    else:  # no dice
                        v = hop_min_v * v_multiplier
                else:
                    v = hop_min_v
                for part, part_i in zip(hop_config, this_i_parts):  # scale i's to be equal
                    part.change_i_v_proposal(v_multiplier * part_i, v)
                hops_i_parts[hop_index] = [i * v_multiplier for i in hops_i_parts[hop_index]]
                hops_v.append(v)  # set v
        else:  # lcm = 0 => at least one hop has 0 current
            # set all hops to 0 current, and intermediate voltages to None (undefined)
            i_values = np.array([sum(i_part) for i_part in hops_i_parts])
            # find locations that leave voltage levels floating: hops with designated 0
            #                                                and hops that can have 0 current with non-0 voltage
            zero_locs = np.flatnonzero((i_values == 0) + np.array(self.hops_can_be_zero))
            if len(zero_locs) == 1:  # only 1 zero current hop
                zero_loc = zero_locs[0]
                hops_v = [0] * len(self.hop_configs)
                hops_v[zero_loc] = randint(hops_min_v[zero_loc], 2*hops_min_v[zero_loc])
            else:
                zero_loc1, zero_loc2 = zero_locs[0], zero_locs[-1]  # first and last
                # between the zero-current parts, v is undefined (np.nan); the nodes are "floating"
                hops_v = [0] * zero_loc1 + [np.nan] * (zero_loc2-zero_loc1+1) + [0] * (len(self.hop_configs)-zero_loc2-1)
            for hop_index, hop_config in enumerate(self.hop_configs):  # set all currents to 0
                for part in hop_config:
                    assert part.i_acceptable(0, v=hops_v[hop_index]), f"Something has gone wrong; {hops_v[hop_index]}" \
                                                                      f"is not an acceptable current for subgraph {self}"
                    part.change_i_v_proposal(0, hops_v[hop_index])
                hops_i_parts[hop_index] = [0 for i in hops_i_parts[hop_index]]
        self.hops_v = hops_v
        self.hops_i_parts = hops_i_parts
        self.hops_undetermined_parts = hops_undetermined_parts
        self.hops_accepted_proposals[(self.proposed_i, sum(hops_v))] = hops_i_parts, hops_v

    def finalize_v_i(self, v=None):
        # everything is already in the parts objects (both type Element and Metagraph), so just need to pass through
        #     for Element.finalize_v_i to resolve element parameters
        if logging.getLogger().isEnabledFor(logging.DEBUG) and v is not None:
            assert v == sum(self.hops_v) or any([np.isnan(float(hop_v)) for hop_v in self.hops_v])
        hops_v = self.hops_v.copy()
        if (v is None or not np.isnan(float(v))) and any([np.isnan(float(hop_v)) for hop_v in self.hops_v]):
            # we are in the highest recursion layer that contains nans
            if v is None:  # only happens at top level, so choose a v-level
                v = randint(np.nansum(hops_v)+2, np.nansum(hops_v)+30)
            nan_locs = np.flatnonzero(np.isnan(hops_v))
            nan_loc1, nan_loc2 = nan_locs[0], nan_locs[-1]  # there are always minimum 2 nans => nan_loc2 > nan_loc1
            hops_v[nan_loc1] = random.randint(0,v)  # inject artificial voltages to ensure fitting parameters
            hops_v[nan_loc2] = v - hops_v[nan_loc1]
            self.max_v_for_floating_part = v  # register in self
        for hop_config, hop_v in zip(self.hop_configs, hops_v):
            for part in hop_config:
                part.finalize_v_i(v=hop_v)

    def get_voltages_dict(self):
        # voltages get returned as dict[(node_from, node_to)] = voltage between node_from, node_to
        # absolute voltage levels wrt. 0V at circuit.end get calculated in Circuit._populate
        result = {}
        for i, hop_config in enumerate(self.hop_configs):
            result[(self.path[i], self.path[i+1])] = self.hops_v[i]
            for part in hop_config:
                if isinstance(part, Metagraph):
                    additional_vs = part.get_voltages_dict()
                    for index, v in additional_vs.items():
                        if index not in result:
                            result[index] = v
                        elif logging.getLogger().isEnabledFor(logging.DEBUG):
                            assert v == result[index]
        return result

    def draw(self, with_solution=False, top_metagraph=True):
        """
        :param with_solution: if True, the current and voltage levels get added to the image
        :return: 5-tuple: Image object, x_start at top, x_end at top, x_start at bottom, x_end at bottom
                 where x_start, x_end are the x-coordinates between which the top resp. bottom node reaches
        """
        hop_images = []
        for hop_index, hop_config in enumerate(self.hop_configs):
            hop_image = make_hop_image([part.draw(with_solution=with_solution, top_metagraph=False)
                                        for part in hop_config])
            hop_img, hop_x_start_top, hop_x_end_top, hop_x_start_bottom, hop_x_end_bottom = hop_image  # unpack
            if with_solution:
                if hop_index > 0 or top_metagraph:
                    first_node = self.path[hop_index]
                    text_width, text_height = first_node.solution_bbox()
                    width = max(hop_img.width, hop_x_end_top + 80 + text_width)  # 40 = 10 before, 30 after
                    height = hop_img.height  # + text_height
                    img = Image.new("RGB", (width, height), (255, 255, 255))
                    img.paste(hop_img, box=(0, 0))
                    draw = ImageDraw.Draw(img)
                    draw.ink = 0
                    draw.text((hop_x_end_top + 20, 0), first_node.solution_text, font=info_font, fill=solution_color)
                    hop_image = img, hop_x_start_top, hop_x_end_top, hop_x_start_bottom, hop_x_end_bottom
            hop_images.append(hop_image)
        if top_metagraph and with_solution:  # make explicit that V(self.end)=0V, add this as an additional "hop image"
            zerov_width, zerov_height = self.path[-1].solution_bbox()
            img = Image.new("RGB", (zerov_width + 180, zerov_height), (255, 255, 255))
            draw = ImageDraw.Draw(img)
            draw.ink = 0
            draw.line((100, 0, 100, zerov_height), width=standard_width)
            draw.text((120, 0), self.path[-1].solution_text, font=info_font, fill=solution_color)
            zerov_hop = img, 100, 100, 100, 100
            hop_images.append(zerov_hop)
        return make_col_image(hop_images)