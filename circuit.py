import random
import logging
import elements
import numpy as np
from random import randint
from fractions import Fraction
from collections import defaultdict
from PIL import Image, ImageDraw
from circuit_parts import Node, Metagraph
from utils import pprint_value, get_divisors, make_col_image, name_font, standard_width, CircuitState

class Circuit:
    def __init__(self, circuitstate=None, **kwargs):
        if circuitstate is None:
            circuitstate = CircuitState(**kwargs)
        self.circuitstate = circuitstate
        self.circuitstate.initialize()
        # some memoization bookkeeping
        self._memoized = {}
        all_elements = 0
        min_elements, max_elements = circuitstate.min_elements, circuitstate.max_elements
        #while all_elements < max(2, min_nodes - 1):
        num_elements = {k: randint(min_elements[k], max_elements[k]) for k in max_elements.keys()}
        #    all_elements = sum(num_elements.values())
        # max_nodes = min(max_nodes, all_elements + 1)  # not more points than series of all elements
        # num_nodes = randint(min_nodes, max_nodes) if max_nodes >= min_nodes else max_nodes
        max_name_length = max([len(el) for el in num_elements.keys()])
        num_elements_print = '\n'.join([f"\t{el: >{max_name_length}} : {num_el}" for el, num_el in num_elements.items()])
        logging.info(f"Generating circuit; number of elements:\n{num_elements_print}")
        # now build the circuit
        self._build_topology(num_elements, 0)  # num_nodes)
        self._populate()  # this also generates self.metagraph

    def _build_topology(self, num_elements, num_nodes):
        # reset counters
        Node.count = 0
        elements.Element.counts = {}
        element_stack = sum([[k] * v for k, v in num_elements.items()], start=[])
        random.shuffle(element_stack)
        # first & last node: start resp. end of voltage source
        self.start, self.end = Node(circuit=self), Node(circuit=self)
        nodes_left = num_nodes - 2
        self.nodes = [self.start, self.end]
        new_element_type = element_stack.pop()
        new_element = getattr(elements, new_element_type)(self.start, self.end)
        self.start.departures.append(new_element)
        self.end.arrivals.append(new_element)
        self.elements = [new_element]
        initial_stack_size = len(element_stack)
        if initial_stack_size <= 1:
            initial_stack_size = 1.0001
        prior_series, prior_parallel = defaultdict(int), defaultdict(int)
        prob_series_at_start, prob_series_at_end = .8, .2 #.8, .2  # in the beginning favour series, towards the end parallel
        prob_series = \
            lambda x: (prob_series_at_start-prob_series_at_end) / (initial_stack_size-1) * (x-1) + prob_series_at_end
        while element_stack:
            # pick an existing element
            element = random.choice(self.elements)
            # choose to add the new element in parallel or series
            # probability of series placement directly proportional to nodes_left / len(element_stack)
            if random.random() <= prob_series(len(element_stack)):  #/initial_stack_size:  #nodes_left / len(element_stack):  # series
                element = random.choices(self.elements,
                                         weights=[1/prior_series[el] if prior_series[el] > 0 else 3
                                                  for el in self.elements], k=1)[0]  # choose "less series" elements more
                nn = Node(circuit=self)
                nodes_left -= 1
                self.nodes.append(nn)
                new_element_type = element_stack.pop()
                new_element = getattr(elements, new_element_type)(nn, element.conn_to)
                element.conn_to = nn
                # node bookkeeping
                nn.arrivals, nn.departures = [element], [new_element]
                new_element.conn_to.arrivals.remove(element)
                new_element.conn_to.arrivals.append(new_element)
                prior_series[element] += 1
                prior_series[new_element] = 1
                while len(element.conn_from.arrivals) == 1:  # add to serial count
                    element = element.conn_from.arrivals[0]
                    prior_series[element] += 1
                while len(new_element.conn_to.departures) == 1:
                    new_element = new_element.conn_to.departures[0]
                    prior_series[new_element] += 1
                prior_parallel[element] = 0  # reduce parallel counts
                for other_element in self.elements:
                    if other_element.conn_from == element.conn_from and other_element.conn_to == new_element.conn_to:
                        prior_parallel[other_element] -= 1
            else:  # parallel
                element = random.choices(self.elements,
                                         weights=[1 / prior_parallel[el] if prior_parallel[el] > 0 else 3
                                                  for el in self.elements], k=1)[0]  # choose "less parallel" elements more
                new_element_type = element_stack.pop()
                new_element = getattr(elements, new_element_type)(element.conn_from, element.conn_to)
                # node bookkeeping
                element.conn_from.departures.append(new_element)
                element.conn_to.arrivals.append(new_element)
                for other_element in self.elements:  # add to parallel counts
                    if other_element.conn_from == element.conn_from and other_element.conn_to == element.conn_to:
                        prior_parallel[element] += 1
                prior_parallel[new_element] = prior_parallel[element]
                # fully recalculate serial counts
                serial_up, serial_up_count = [element], 0
                while len(serial_up[-1].conn_from.arrivals) == 1:
                    serial_up.append(serial_up[-1].conn_from.arrivals[0])
                    serial_up_count += 1
                if serial_up_count > 1:
                    serial_up_count -= 1  # we counted element, which is no longer in the series
                    for up_element in serial_up[1:]:
                        prior_series[up_element] = serial_up_count
                serial_down, serial_down_count = [element], 0
                while len(serial_down[-1].conn_to.departures) == 1:
                    serial_down.append(serial_down[-1].conn_to.departures[0])
                    serial_down_count += 1
                if serial_down_count > 1:
                    serial_down_count -= 1  # we counted element, which is no longer in the series
                    for down_element in serial_down[1:]:
                        prior_series[down_element] = serial_down_count
            self.elements.append(new_element)
        pass

    def _renumber_nodes(self, force=False):
        # renumber to make numbering more logical: every incoming connection at a node comes from nodes before it
        if "renumber" not in self._memoized or self._memoized["renumber"] is False or force:
            number = 0
            renumber_q = [self.start]
            renumber_done = set()
            while renumber_q:
                next_node = renumber_q.pop(0)
                if next_node not in renumber_done:
                    # all arrivals need to have been numbered
                    assert all([el.conn_from in renumber_done for el in next_node.arrivals])
                    next_node.number = number
                    number += 1
                    renumber_done.add(next_node)
                    renumber_q.extend([el.conn_to for el in next_node.departures])
                    renumber_q.sort(key=lambda n: sum([el.conn_from not in renumber_done for el in n.arrivals]))
            self.nodes.sort(key=lambda n:n.number)
            self._memoized["renumber"] = True

    def _layer_layout(self, force=False):
        """
        :return: dict: node -> layer height that node is at
        The final drawing = layers of node-to-node parts where some parts can span multiple layers
        Convention: node layer gets numbering of the parts layer below
        """
        if "layer_layout" not in self._memoized or force:
            # first make a dict: node -> list of all nodes that follow that node
            self._renumber_nodes()  # necessary order.
            following = {self.end: set()}
            for n in self.nodes[-2::-1]:
                direct_follow = set([el.conn_to for el in n.departures])
                following[n] = direct_follow.union(*[following[dfn] for dfn in direct_follow])
            stack = {self.start}
            node_layout = {self.start: 0}
            layer_count = 1
            while stack:
                candidate_layer = set().union(*[set([el.conn_to for el in n.departures]) for n in stack])
                # additional requirement: the node can not follow any node that is in next_layer
                following_next = set().union(*[following[n]for n in candidate_layer])
                next_layer = candidate_layer.difference(following_next)
                for n in next_layer:
                    node_layout[n] = layer_count
                layer_count += 1
                stack = next_layer
            self._memoized["layer_layout"] = node_layout
        return self._memoized["layer_layout"]


    def _populate(self):
        # method that seeks element, voltage and current levels that are 'easily calculable'
        # First calculate the meta-graph and determine a current step
        layer_layout = self._layer_layout()
        self.metagraph = Metagraph(self)
        metagraph = self.metagraph  # too lazy to type self
        # determining the current step
        milli, micro, nano = Fraction(1, 10**3), Fraction(1, 10**6), Fraction(1, 10**9)
        steps_per_3orders = [1, 2, 5, 10, 25, 50, 100, 250, 500]
        current_steps = [step * order for step in steps_per_3orders for order in (milli, micro, nano)]
        threshold = 1 / 10**(len(self.elements)/len(self.nodes))  # the more elements per node, the smaller max steps
        current_step = random.choice([step for step in current_steps if step < threshold])
        istep_per_vstep = self.circuitstate.voltage_step / current_step
        if istep_per_vstep.denominator > 1:  # it is required that vstep is divisible by current_step
            self.circuitstate.voltage_step *= istep_per_vstep.denominator
            istep_per_vstep = istep_per_vstep.numerator
        # precompute divisors
        istep_per_vstep_divisors = get_divisors(istep_per_vstep)
        # save in self for consultation by self.metagraph and Elements
        self.istep = current_step
        self.istep_per_vstep = istep_per_vstep
        self.istep_per_vstep_divisors = istep_per_vstep_divisors

        # To do the actual assignment of voltages and currents, we will largely make use of the recursion that
        # the Metagraph object (with methods mirrored in Element subclasses) allows. Anchor point is metagraph.get_i_v.
        # This method does largely the following:
        #     - Generating voltage and current proposals for all elements
        #     - At several point in the process, perform 'arbitrage' to seek for values that will allow for a 'nicer'
        #            voltages and total currents
        metagraph.get_i_v()

        # Once 'the best' candidates are found for the full circuit, make them definitive, while asking the elements to
        #     calculate parameters (such as R, Vf, ...) that are compatible with the given values. Also, since all
        #     calculations up to now have been in v- and i-steps, convert into the proper voltages and currents.
        metagraph.finalize_v_i()

        # We have voltages over the edges, now translate that in absolute levels vs. 0V at self.end
        edge_voltages = metagraph.get_voltages_dict()
        for i in edge_voltages.keys():  # convert to proper voltages
            edge_voltages[i] *= self.circuitstate.voltage_step
        self.end.voltage = 0
        stack = [self.end]
        done = set()
        if logging.getLogger().isEnabledFor(logging.DEBUG):  # extra debug checks, 'precompute' the level check
            def debug_check(node, v): assert node.voltage == v or np.isnan(float(v)), "Error in voltage assignment"
        else:
            def debug_check(node, v): pass
        while stack:
            node_to = stack.pop(0)
            for node_from in set([el.conn_from for el in node_to.arrivals]):
                if node_from in done:
                    debug_check(node_from, node_to.voltage + edge_voltages[(node_from, node_to)])
                else:
                    edge_v = edge_voltages[(node_from, node_to)]
                    if not np.isnan(float(edge_v)):
                        node_from.voltage = node_to.voltage + edge_v
                        stack.append(node_from)
            done.add(node_to)
        if any([node.voltage is None for node in self.nodes]):  # redo from other direction
            if self.start.voltage is None:  # no way to infer start voltage except <= metagraph.max_v_for_floating_part
                self.start.voltage = random.randint(1, metagraph.max_v_for_floating_part) * self.circuitstate.voltage_step
            stack = [self.start]
            done = set()
            if logging.getLogger().isEnabledFor(logging.DEBUG):  # extra debug checks, 'precompute' the level check
                def debug_check(node, v):
                    assert node.voltage == v or np.isnan(float(v)), "Error in voltage assignment"
            else:
                def debug_check(node, v):
                    pass
            while stack:
                node_from = stack.pop(0)
                for node_to in set([el.conn_to for el in node_from.departures]):
                    if node_to in done:
                        debug_check(node_to, node_from.voltage - edge_voltages[(node_from, node_to)])
                    else:
                        edge_v = edge_voltages[(node_from, node_to)]
                        if not np.isnan(float(edge_v)):
                            if node_to.voltage is not None:
                                debug_check(node_to, node_from.voltage - edge_v)
                            node_to.voltage = node_from.voltage - edge_v
                            stack.append(node_to)
                done.add(node_from)

    def netlist(self):
        result = [".title netlist generated by CircuitPractice",
                  "V1 {} {} {}".format(self.start.netlist_repr(), self.end.netlist_repr(), self.start.voltage)]
        all_elements = self.elements.copy()
        all_elements.sort(key=lambda e: len(self.nodes) * e.conn_from.number + e.conn_to.number)
        for el in all_elements:
            result.append(el.netlist_repr())
        result.append(".end")
        return result

    def solution(self):
        i_V = sum(el.i for el in self.nodes[0].departures)
        result = ["I(V1): \t-" + pprint_value(i_V) + 'A']
        for el in self.elements:
            result.append("I({}): \t".format(el.name) + pprint_value(el.i) + 'A')
        for n in self.nodes:
            result.append("V({}): \t".format(n.netlist_repr()) + pprint_value(n.voltage) + 'V')
        return result

    def draw(self, with_solution=False):
        """Draw the circuit. Either show on screen or save to file <filename>"""
        # everything comes recursively form metagraph.draw, which breaks down circuit in simpler parts
        final_img, final_x_start_top, final_x_end_top, final_x_start_bottom, final_x_end_bottom = \
            self.metagraph.draw(with_solution=with_solution)
        # now draw battery and connect
        width_add = 1000
        width, height = final_img.size[0] + width_add, final_img.size[1] + 280
        full_img = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(full_img)
        V = '?' if self.start.voltage is None else pprint_value(self.start.voltage) + 'V'
        textbbox = draw.multiline_textbbox((0, 0), f"V1\n{V}", font=name_font)
        if textbbox[2] > width_add - 500:  # I don't expect this to happen, but better safe than sorry
            width_add = 500 + textbbox[2]
            width = final_img.size[0] + width_add
            full_img = Image.new("RGB", (width, height), (255, 255, 255))
            draw = ImageDraw.Draw(full_img)
        draw.ink = 0
        full_img.paste(final_img, box=(width_add, 140))
        # voltage source = circle + -
        draw.arc((40, height//2 - 200, 440, height//2 + 200), 0, 720, width=standard_width)  # circle
        draw.line((200, height//2 - 120, 280, height//2 - 120), width=standard_width)  # +
        draw.line((240, height//2 - 160, 240, height//2 - 80), width=standard_width)
        draw.line((200, height//2 + 120, 280, height//2 + 120), width=standard_width)  # -
        draw.multiline_text((460, (height - textbbox[3]) // 2), f"V1\n{V}", font=name_font)
        # lines to beginning
        draw.line((240, 40, 240, height//2 - 200), width=standard_width)
        draw.line((240, 40, width_add + (final_x_end_top + final_x_start_top) // 2, 40), width=standard_width)
        draw.line((width_add + (final_x_end_top + final_x_start_top) // 2, 40,
                   width_add + (final_x_end_top + final_x_start_top) // 2, 140), width=standard_width)
        # lines to end
        draw.line((240, height//2 + 200, 240, height - 40), width=standard_width)
        draw.line((240, height - 40, width_add + (final_x_end_bottom + final_x_start_bottom) // 2, height - 40),
                  width=standard_width)
        draw.line((width_add + (final_x_end_bottom + final_x_start_bottom) // 2, height - 140,
                   width_add + (final_x_end_bottom + final_x_start_bottom) // 2, height - 40), width=standard_width)
        return full_img
