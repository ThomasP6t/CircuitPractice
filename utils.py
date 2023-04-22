import random
import logging
import os
from fractions import Fraction
from decimal import Decimal

import numpy as np
from PIL import Image, ImageDraw, ImageFont

version = '1.0'

# fonts are from https://www.fontspace.com
name_font = ImageFont.truetype("Lmmono12Regular.otf", 120)
info_font = ImageFont.truetype("Lmmono12Regular.otf", 80)
extra_font = ImageFont.truetype("Lmmono12Regular.otf", 46)
standard_width = 5
solution_color = (255, 0, 255)


# helper function for value printing
def pprint_value(value):
    if type(value) is bool:
        return str(value)
    elif hasattr(value, "__float__") and np.isnan(float(value)):
        return "NA"
    powers_of_10 = {6: 'meg', 3: 'k', 0: '', -3: 'm', -6: 'u', -9: 'n'}
    if value == 0:
        return '0'
    if isinstance(value, Fraction):
        value = value.numerator / Decimal(value.denominator)
    else:
        try:
            value = Decimal(value)
        except Exception as e:
            logging.debug(f"Casting value {value} as Decimal via {str(value)}")
            value = Decimal(str(value))
    no_decimals, one_decimal = Decimal("1"), Decimal("1.0")
    for p in powers_of_10.keys():
        if p > 0:
            if value % 10 ** p == 0:
                str_value = str((value / 10 ** p).quantize(no_decimals))
                return str_value + powers_of_10[p]
            elif value % 10 ** (p-1) == 0 and value >= 10 ** p:
                str_value = str((value / 10 ** p).quantize(one_decimal))
                return str_value + powers_of_10[p]
        elif p == 0:
            if value % 1 == 0:
                str_value = str(value.quantize(no_decimals))
                return str_value
            elif value * 10 % 1 == 0 and value > 1:
                str_value = str(value.quantize(one_decimal))
                return str_value
        else:  # p < 0
            if value * 10 ** (-p) % 1 == 0:
                str_value = str((value * 10**(-p)).quantize(no_decimals))
                return str_value + powers_of_10[p]
            elif value * 10 ** (-p + 1) % 1 == 0 and value > 10 ** p:
                str_value = str((value * 10**(-p)).quantize(one_decimal))
                return str_value + powers_of_10[p]
    return str(value)


def get_divisors(number):
    if number == 1:  # corner case
        return [1]
    rest = number
    i = 2
    result = []
    while rest > 1:
        if rest % i == 0:
            result.append(i)
            rest /= i
        else:
            i += 1
    return result


def partition(vars_can_be_zero, total):
    # generator that gives all partitions (distribution of total to the vars)
    if len(vars_can_be_zero) == 1:
        if vars_can_be_zero[0] or total > 0:
            yield (total,)
        else:
            return
    else:
        start = 0 if vars_can_be_zero[0] else 1
        for value in range(start, total + 1):
            for permutation in partition(vars_can_be_zero[1:], total - value):
                yield (value,) + permutation


def weighted_sample(pop, weights, k):
    # Get k elements from pop, with probability ~ weight
    # From https://utopia.duth.gr/~pefraimi/research/data/2007EncOfAlg.pdf
    assert len(pop) == len(weights), "Weights needs to be of the same length as pop"
    scores = np.log(np.random.random(len(pop))) / np.array(weights)
    top_k_locs = sorted(range(len(pop)), key=lambda i: scores[i], reverse=True)[:k]
    return np.array(pop)[top_k_locs]


def make_col_image(imgs):
    """
    helper function that makes a column of parts ('hops'), as used in Metagraph.draw
    :param imgs: list of tuples (Image, x_start_top, x_end_top, x_start_bottom, x_end_bottom)
    :return: one tuple of the same format, stacking vertically all parts
    """
    col_img, col_x_start_top, col_x_end_top, col_x_start_bottom, col_x_end_bottom = imgs[0]  # unpack
    for img in imgs[1:]:
        new_img, new_x_start_top, new_x_end_top, new_x_start_bottom, new_x_end_bottom = img  # unpack
        # define an offset of new_img wrt. col_img
        offset = (col_img.size[0] - new_img.size[0]) // 2
        # and require some overlap in interval x_start - x_end
        if col_x_end_bottom < offset + new_x_start_top:
            offset = col_x_end_bottom - new_x_start_bottom
        elif col_x_start_bottom > offset + new_x_end_top:
            offset = col_x_start_bottom - new_x_end_top
        new_width = max(col_img.size[0], offset + new_img.size[0]) - min(0, offset)
        new_height = col_img.size[1] + new_img.size[1] - standard_width
        col_offset, new_offset = max(0, -offset), max(0, offset)  # split the offset
        new_col_img = Image.new("RGB", (new_width, new_height), (255, 255, 255))
        new_col_img.paste(col_img, box=(col_offset, 0))
        new_col_img.paste(new_img, box=(new_offset, col_img.size[1] - standard_width))
        draw = ImageDraw.Draw(new_col_img)  # need to redraw lower line of col_img
        draw.ink = 0
        draw.line((col_offset + col_x_start_bottom, col_img.size[1] - standard_width//2 - 1,
                   col_offset + col_x_end_bottom, col_img.size[1] - standard_width//2 - 1),
                  width=standard_width)
        col_img = new_col_img
        col_x_start_top += col_offset
        col_x_end_top += col_offset
        col_x_start_bottom = new_x_start_bottom + new_offset
        col_x_end_bottom = new_x_end_bottom + new_offset
    return col_img, col_x_start_top, col_x_end_top, col_x_start_bottom, col_x_end_bottom


def make_hop_image(imgs):
    """
    helper function that makes a row of parts, as used in Metagraph.draw for within a hop
    :param imgs: list of tuples (Image, x_start_top, x_end_top, x_start_bottom, x_end_bottom)
    :return: one tuple of the same format, stacking horizontally all parts
    """
    width = sum([img[0].size[0] for img in imgs])
    height = max([img[0].size[1] for img in imgs])
    # deal with shorter columns first:
    for i, img in enumerate(imgs):
        if img[0].size[1] < height:
            col_img, col_x_start_top, col_x_end_top, col_x_start_bottom, col_x_end_bottom = img  # unpack
            new_img = Image.new("RGB", (col_img.size[0], height), (255, 255, 255))
            y_offset = (height - col_img.size[1]) // 2
            new_img.paste(col_img, box=(0, y_offset))
            draw = ImageDraw.Draw(new_img)
            draw.ink = 0
            top_x = (col_x_start_top + col_x_end_top) // 2
            bottom_x = (col_x_start_bottom + col_x_end_bottom) // 2
            draw.line((top_x, 0, top_x, y_offset), width=standard_width)
            draw.line((bottom_x, y_offset + col_img.size[1], bottom_x, height), width=standard_width)
            imgs[i] = new_img, top_x, top_x, bottom_x, bottom_x
    full_img = Image.new("RGB", (width, height), (255, 255, 255))
    x = 0
    for img in imgs:
        full_img.paste(img[0], box=(x, 0))
        x += img[0].size[0]
    full_x_start_top, full_x_end_top = imgs[0][1], width - (img[0].size[0] - img[2])
    full_x_start_bottom, full_x_end_bottom = imgs[0][3], width - (img[0].size[0] - img[4])
    draw = ImageDraw.Draw(full_img)
    draw.ink = 0
    draw.line((full_x_start_top, standard_width // 2, full_x_end_top, standard_width // 2),
              width=standard_width)
    draw.line((full_x_start_bottom, height - standard_width // 2 - 1,
               full_x_end_bottom, height - standard_width // 2 - 1),
              width=standard_width)
    return full_img, full_x_start_top, full_x_end_top, full_x_start_bottom, full_x_end_bottom


def add_text_to_img(img, text):
    if text:
        draft_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))  # (for bboxes)
        bbox = draft_draw.multiline_textbbox((0, 0), text, font=extra_font, align='left')
        width, height = bbox[2:4]
        new_img = Image.new("RGB", (max(img.width, width+40), img.height + height), (255, 255, 255))
        new_img.paste(img, box=(0, height))
        draw = ImageDraw.Draw(new_img)
        draw.ink = 0
        draw.multiline_text((20, 0), text, font=extra_font, align='left')
        img = new_img
    return img


def make_interpage(text):
    width, height = 1200, 1600
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.ink = 0
    bbox = draw.multiline_textbbox((0, 0), text, font=name_font)
    text_x, text_y = (width - bbox[2]) // 2, (height - bbox[3]) // 2
    draw.multiline_text((text_x, text_y), text, font=name_font)
    return img


solution_interpage = make_interpage("Solutions")


def make_pdf(image_list, file_name, path=None):
    if path:
        file_name = os.path.join(path, file_name)
    image_list[0].save(file_name, "PDF", resolution=100.0, save_all=True, append_images=image_list[1:])


def seeder():
    return random.randint(-10 ** 64, 10 ** 64)


default_min_nodes = 3
default_min_elements = {"Resistor": 2, "Diode": 1, "Zener": 0}
default_max_nodes = 6
default_max_elements = {"Resistor": 5, "Diode": 3, "Zener": 2}
defaults = {"min_elements": default_min_elements.copy,
            "max_elements": default_max_elements.copy,
            "undetermined_prob": 0.75,  # probability of introducing undetermined elements where possible
            #                             see circuit_parts.Metagraph.prime_for_v_i for more info
            "voltage_step": 0.5,
            "seed": seeder}


class CircuitState:
    """
    Simple State keeper for a Circuit object
    """
    def __init__(self, **kwargs):
        # see above defined defaults for possible kwargs
        self.cp_version = version
        logging.info(f"Initiating CircuitState. CircuitPractice v.{version}")
        for prop in defaults.keys():
            if prop not in kwargs or kwargs[prop] is None:
                default_value = defaults[prop]() if callable(defaults[prop]) else defaults[prop]
                if prop != "seed":
                    logging.warning(f"{prop} has not been explicitly defined in CircuitState.\n"
                                    f"\tThis could lead to reproducibility issues between versions."
                                    f"\tSetting to: {default_value}")
                setattr(self, prop, default_value)
            else:
                setattr(self, prop, kwargs[prop])
            if prop == "voltage_step":
                self.voltage_step = Fraction(self.voltage_step).limit_denominator(1000)
        self.parameter_attribs = ["cp_version"]
        self.parameter_attribs.extend(defaults.keys())

    def initialize(self):
        # (re)initialize the random number generator
        logging.info(f"Initializing random number generator.\nseed: {self.seed}")
        random.seed(self.seed)

    def __str__(self):
        result = "CircuitState with parameters:\n"
        max_param_length = max([len(param) for param in self.parameter_attribs])
        result += "\n".join([f"\t{param}: {getattr(self, param): : >{max_param_length}}"
                             for param in self.parameter_attribs])
        return result