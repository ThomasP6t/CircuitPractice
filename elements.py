import itertools
import logging
import random
import numpy as np
from PIL import Image, ImageDraw
from utils import pprint_value, name_font, info_font, standard_width, solution_color


class Element:
    name_char = '?'
    counts = {}
    can_be_undetermined = False  # see circuit_parts.Metagraph.get_i_v for definition
    open_for_i_change = 0  # measure of how easy currents can be 'nudged' while staying 'nice', in % (0<=.<=1)

    def __init__(self, conn_from, conn_to):
        self.conn_from, self.conn_to = conn_from, conn_to
        if self.__class__ in self.counts:
            self.counts[self.__class__] += 1
        else:
            self.counts[self.__class__] = 1
        self.set_number(self.counts[self.__class__])
        self.params = {}
        self.proposed_i = None
        self.i = None
        self.undetermined = False

    def __repr__(self, indent=1):
        return repr(self.conn_from) + '->' + self.name + '->' + repr(self.conn_to)

    def set_number(self, number):
        self.number = number
        self.name = self.name_char + str(self.number)

    @property
    def can_be_zero(self):
        """ simple property that indicates whether the element can have a current of 0 while still having a non-zero
        voltage over it. """
        raise NotImplementedError

    def get_nice_current_parts(self, make_undetermined=False, oom_center=None):
        raise NotImplementedError

    def i_acceptable(self, i, v=None):
        """
        method used in current arbitrage within circuit_parts.Metaclass
        :param i: proposal for a new i, in 'current parts' (=multiples of i_step)
        :param v: voltage over the element, in 'voltage steps'
        :return: bool, True if i can be accepted, False if not
        """
        # standard self.open_for_i_change = 0, so:
        return False  # computer says nooo

    def change_i_v_proposal(self, i, v=None):
        if not self.i_acceptable(i, v):
            v_text = "" if v is None else f", V={pprint_value(v)}V"
        assert self.i_acceptable(i, v),\
            f"I={pprint_value(i)}A{v_text} is not an acceptable value for element {repr(self)}."
        self.proposed_i = i

    def finalize_v_i(self, v):
        # final assignment
        raise NotImplementedError

    # a few properties to get the text bboxes/Images for drawing
    @property
    def name_text(self):
        if "name" in dir(self):
            return f"{self.name}"
        else:
            return f"{self.name_char}?"

    @property
    def info_text(self):
        return "?"

    @property
    def solution_text(self):
        i = '?' if self.i is None else pprint_value(self.i)
        return f"i={i}A"

    def _get_bboxes(self, with_solution=False):
        """
        Helper function that gives text bboxes
        :return: list of bbox (=tuple x_start, y_start, x_end, y_end) for name, info and possibly solution
        """
        draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        bboxes = [draw.textbbox((0, 0), self.name_text, font=name_font),
                  draw.multiline_textbbox((0, 0), self.info_text, font=info_font)]
        if with_solution:
            bboxes.append(draw.textbbox((0, 0), self.solution_text, font=info_font))
        return bboxes

    def text_bbox(self, with_solution=False):
        """
        :return: tuple with width and height of element text
        """
        bboxes = self._get_bboxes(with_solution=with_solution)
        width = max([bbox[2] for bbox in bboxes])
        height = sum([bbox[3] for bbox in bboxes])
        return width, height

    def _generate_imagedraw(self, with_solution=False):
        """
        :return: ImageDraw object with sufficient width for text
        convention: height of image: 320
                    element connection at x=50
                    element symbol does not go further than x=80
        """
        text_width, text_height = self.text_bbox(with_solution=with_solution)
        width = max(400, 240+text_width)  # 120 = 50 line, 40 until text, 30 after text
        height = max(640, text_height + 120)
        img = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.ink = 0
        return draw

    def put_text(self, draw, with_solution=False):
        bboxes = self._get_bboxes(with_solution=with_solution)
        text_cum_heights = np.cumsum([bbox[3] for bbox in bboxes])
        start_y = (draw._image.height - text_cum_heights[-1]) // 2
        draw.text((180, start_y), self.name_text, font=name_font)
        draw.multiline_text((180, start_y+text_cum_heights[0]), self.info_text, font=info_font)
        if with_solution:
            draw.text((180, start_y+text_cum_heights[1]), self.solution_text, font=info_font, fill=solution_color)

    def draw(self, with_solution=False, top_metagraph=False):
        raise NotImplementedError()


class Resistor(Element):
    name_char = 'R'
    open_for_i_change = 0.1

    def netlist_repr(self):
        R = '?' if 'R' not in self.params else pprint_value(self.params['R'])
        return ' '.join([self.name, self.conn_from.netlist_repr(), self.conn_to.netlist_repr(), R])

    @property
    def can_be_zero(self):
        """ simple property that indicates whether the element can have a current of 0 while still having a non-zero
                voltage over it. """
        return False

    def get_nice_current_parts(self, make_undetermined=False, oom_center=None):
        # return a number of current steps that satisfies the condition that these remain 'nice' independent of voltage
        # since all voltages are multiples of vstep, and currents of istep (which divides vstep), per Ohm's law we get:
        #     k_v * vstep/istep = k_i * R  with V = k_v * vstep, I = k_i * istep, k_v, k_i integers > 0
        # so if k_i divides vstep/istep, R is an integer independent of k_v
        # vstep/istep divisors are saved in Circuit object for efficiency reasons
        if make_undetermined:
            raise TypeError("make_undetermined cannot be True for Resistor.")
        # return a divisor of istep_per_vstep
        possible_divisors = sorted(self.conn_from.circuit.istep_per_vstep_divisors)
        if oom_center is None:
            # limit max current to 2/3
            possible_number_of_divs = np.arange(1, int(np.ceil(len(possible_divisors) * 2 / 3)+1))
            number_of_divs = random.choices(possible_number_of_divs, weights=1/possible_number_of_divs**3)[0]
        else:  # rough but non-expensive proxy of acceptable range of number of divisors to allow
            from_smallest_divs = np.cumprod(possible_divisors)
            from_largest_divs = np.cumprod(possible_divisors[::-1])
            smallest_number_of_divs = np.flatnonzero(from_largest_divs >= oom_center/10)[0] + 1
            largest_number_of_divs = np.flatnonzero(from_smallest_divs <= oom_center*10)[-1] + 1
            possible_number_of_divs = np.arange(smallest_number_of_divs, largest_number_of_divs+1)
            number_of_divs = random.choices(possible_number_of_divs, weights=1/possible_number_of_divs**3)[0]
        chosen_divs = random.sample(possible_divisors, number_of_divs)
        self.proposed_i = np.prod(chosen_divs)
        return self.proposed_i

    def i_acceptable(self, i, v=None):
        if v is None:
            if i == 0:
                return False
            return self.conn_from.circuit.istep_per_vstep % i == 0
        else:
            if v == 0 or np.isnan(float(v)):
                return i == 0
            else:
                return i > 0 and ((v * self.conn_from.circuit.istep_per_vstep) % i == 0)

    def finalize_v_i(self, v):
        # final assignment
        self.i = self.proposed_i * self.conn_from.circuit.istep
        v *= self.conn_from.circuit.circuitstate.voltage_step
        if self.i > 0:
            R = v / self.i
        else:  # i == 0
            R = random.randint(1,9)
            R *= 10 ** random.randint(1, 8)
        if logging.getLogger().isEnabledFor(logging.DEBUG):  # extra debug checks
            if int(R) != R:
                logging.warning(f"Non-integer value of R in Resistor {repr(self)}: {R}")
        self.params['R'] = R

    @property
    def info_text(self):
        return '?' if 'R' not in self.params else pprint_value(self.params['R'])

    def draw(self, with_solution=False, top_metagraph=False):
        draw = self._generate_imagedraw(with_solution=with_solution)
        symbol_y_shift = (draw._image.height - 640) // 2
        draw.line((100, 0, 100, 120+symbol_y_shift), width=standard_width)
        draw.rectangle((40, 120+symbol_y_shift, 160, 540+symbol_y_shift), width=standard_width)
        draw.line((100, 540+symbol_y_shift, 100, draw._image.height), width=standard_width)
        self.put_text(draw, with_solution=with_solution)
        return draw._image, 100, 100, 100, 100


class Diode(Element):
    """A near ideal diode, with a (possibly) non-zero forward voltage, at which it conducts perfectly"""
    name_char = 'D'
    can_be_undetermined = True

    def __init__(self, conn_from, conn_to):
        super().__init__(conn_from, conn_to)
        self.undetermined = None  # need to run get_nice_current_parts first

    def netlist_repr(self):
        reverse = False if 'reverse' not in self.params else self.params['reverse']
        netlist_from, netlist_to = (self.conn_to, self.conn_from) if reverse else (self.conn_from, self.conn_to)
        return ' '.join([self.name, netlist_from.netlist_repr(), netlist_to.netlist_repr()] + self.info_text.split('\n'))

    @property
    def can_be_zero(self):
        """ simple property that indicates whether the element can have a current of 0 while still having a non-zero
                voltage over it. """
        return True

    @property
    def open_for_i_change(self):
        return float(self.undetermined)

    def get_nice_current_parts(self, make_undetermined=False, oom_center=None):
        # return a number of current steps that satisfies the condition that these remain 'nice' independent of voltage
        # for Diodes: always the case
        #     - if determined (= zero current), just need to set Vf (or Vz for subclass Zener) > V
        #     - if undetermined, V == Vf (or Vz), so just have to set this.
        if make_undetermined:
            self.undetermined = True
            possible_divisors = sorted(self.conn_from.circuit.istep_per_vstep_divisors)
            if oom_center is None:
                # also return a divisor of istep_per_vstep, but higher chance of less divs
                # limit max current to 2/3
                possible_number_of_divs = np.arange(1, int(np.ceil(len(possible_divisors) * 2 / 3)+1))
                number_of_divs = random.choices(possible_number_of_divs, weights=1/possible_number_of_divs**4)[0]
            else:  # rough but non-expensive proxy of acceptable range of number of divisors to allow
                from_smallest_divs = np.cumprod(possible_divisors)
                from_largest_divs = np.cumprod(possible_divisors[::-1])
                smallest_number_of_divs = np.flatnonzero(from_largest_divs >= oom_center / 10)[0] + 1
                largest_number_of_divs = np.flatnonzero(from_smallest_divs <= oom_center * 10)[-1] + 1
                possible_number_of_divs = np.arange(smallest_number_of_divs, largest_number_of_divs + 1)
                number_of_divs = random.choices(possible_number_of_divs, weights=1/possible_number_of_divs**4)[0]
            chosen_divs = random.sample(possible_divisors, number_of_divs)
            self.proposed_i = np.prod(chosen_divs)
            return self.proposed_i
        self.undetermined = False
        self.proposed_i = 0
        return 0

    def i_acceptable(self, i, v=None):
        return True  # anything goes for this modelling of diodes

    def make_random_Vf(self, over=None):
        if over is None:
            potential_Vf = np.arange(2, 21) * self.conn_from.circuit.circuitstate.voltage_step
        else:
            potential_Vf = np.arange(over+1, max(2*over, over+10))
        return random.choices(potential_Vf, weights=1/np.arange(2, 2+len(potential_Vf)), k=1)[0]

    def finalize_v_i(self, v):
        # final assignment
        self.i = self.proposed_i * self.conn_from.circuit.istep
        self.params['reverse'] = False
        v *= self.conn_from.circuit.circuitstate.voltage_step
        if self.i > 0:
            self.params['Vf'] = v
        else:
            # either reversed or a too high Vf
            if hasattr(self, "undetermined") and self.undetermined and not np.isnan(float(v)):
                self.params['Vf'] = v
            elif random.random() < .6:  # reverse with 60% chance
                self.params['reverse'] = True
                self.params['Vf'] = self.make_random_Vf()
            else:
                if np.isnan(float(v)):
                    self.params['Vf'] = self.make_random_Vf()
                else:
                    self.params['Vf'] = self.make_random_Vf(v)

    @property
    def info_text(self):
        Vf = '?' if 'Vf' not in self.params else pprint_value(self.params['Vf']) + 'V'
        return f"Vf={Vf}"

    def draw(self, with_solution=False, top_metagraph=False):
        draw = self._generate_imagedraw(with_solution=with_solution)
        symbol_y_shift = (draw._image.height - 640) // 2
        draw.line((100, 0, 100, 280+symbol_y_shift), width=standard_width)
        draw.line((100, 380+symbol_y_shift, 100, draw._image.height), width=standard_width)
        draw.line((40, 280+symbol_y_shift, 160, 280+symbol_y_shift), width=standard_width)
        draw.line((40, 380+symbol_y_shift, 160, 380+symbol_y_shift), width=standard_width)
        if 'reverse' in self.params and self.params['reverse']:
            draw.line((100, 280+symbol_y_shift, 40, 380+symbol_y_shift), width=standard_width)
            draw.line((100, 280+symbol_y_shift, 160, 380+symbol_y_shift), width=standard_width)
        else:
            draw.line((100, 380+symbol_y_shift, 40, 280+symbol_y_shift), width=standard_width)
            draw.line((100, 380+symbol_y_shift, 160, 280+symbol_y_shift), width=standard_width)
        self.put_text(draw, with_solution=with_solution)
        return draw._image, 100, 100, 100, 100


class Zener(Diode):
    """A near ideal Zener diode, with
           a (possibly) non-zero forward voltage, at which it conducts perfectly in the forward direction
           a non-zero zener voltage, at which it conducts perfectly in the reverse direction"""
    name_char = 'Z'

    def make_random_Vz(self, over=None):
        if over is None:
            potential_Vz = np.arange(3, 33) * self.conn_from.circuit.circuitstate.voltage_step
        else:
            potential_Vz = np.arange(over+1, (over+1) + 4*(over+1)**.5)
        return random.choices(potential_Vz, weights=1/np.arange(6, 6+len(potential_Vz)), k=1)[0]

    def finalize_v_i(self, v):
        # final assignment
        self.i = self.proposed_i * self.conn_from.circuit.istep
        v *= self.conn_from.circuit.circuitstate.voltage_step
        reverse = True if random.random() < .8 else False  # Zeners are used more often in a reversed state
        if self.i == 0 and v == 0:  # avoid Vz = 0
            reverse = False
        self.params['reverse'] = reverse
        if np.isnan(float(v)):
            self.params['Vf'] = self.make_random_Vf()
            self.params['Vz'] = self.make_random_Vz()
        elif self.i > 0:
            if reverse:
                self.params['Vf'] = self.make_random_Vf()
                self.params['Vz'] = v
            else:
                self.params['Vf'] = v
                self.params['Vz'] = self.make_random_Vz()
        else:
            # too high Vf or Vz
            if reverse:
                self.params['Vf'] = self.make_random_Vf()
                if hasattr(self, "undetermined") and self.undetermined:
                    self.params['Vz'] = v
                else:
                    self.params['Vz'] = self.make_random_Vz(v)
            else:
                if hasattr(self, "undetermined") and self.undetermined:
                    self.params['Vf'] = v
                else:
                    self.params['Vf'] = self.make_random_Vf(v)
                self.params['Vz'] = self.make_random_Vz()

    @property
    def info_text(self):
        Vf = '?' if 'Vf' not in self.params else pprint_value(self.params['Vf']) + 'V'
        Vz = '?' if 'Vz' not in self.params else pprint_value(self.params['Vz']) + 'V'
        return f"Vf={Vf}\nVz={Vz}"

    def draw(self, with_solution=False, top_metagraph=False):
        draw = self._generate_imagedraw(with_solution=with_solution)
        symbol_y_shift = (draw._image.height - 640) // 2
        draw.line((100, 0, 100, 280+symbol_y_shift), width=standard_width)
        draw.line((100, 380+symbol_y_shift, 100, draw._image.height), width=standard_width)
        draw.line((40, 280+symbol_y_shift, 160, 280+symbol_y_shift), width=standard_width)
        draw.line((40, 380+symbol_y_shift, 160, 380+symbol_y_shift), width=standard_width)
        if 'reverse' in self.params and self.params['reverse']:
            draw.line((100, 280+symbol_y_shift, 40, 380+symbol_y_shift), width=standard_width)
            draw.line((100, 280+symbol_y_shift, 160, 380+symbol_y_shift), width=standard_width)
            draw.line((40, 250+symbol_y_shift, 40, 280+symbol_y_shift), width=standard_width)
            draw.line((160, 280+symbol_y_shift, 160, 310+symbol_y_shift), width=standard_width)
        else:
            draw.line((100, 380+symbol_y_shift, 40, 280+symbol_y_shift), width=standard_width)
            draw.line((100, 380+symbol_y_shift, 160, 280+symbol_y_shift), width=standard_width)
            draw.line((40, 350+symbol_y_shift, 40, 380+symbol_y_shift), width=standard_width)
            draw.line((160, 380+symbol_y_shift, 160, 410+symbol_y_shift), width=standard_width)
        self.put_text(draw, with_solution=with_solution)
        return draw._image, 100, 100, 100, 100
