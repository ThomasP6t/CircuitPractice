# CircuitPractice
import logging
import random
from circuit import Circuit
from utils import CircuitState, make_pdf, seeder, version, solution_interpage, add_text_to_img


def make_exercise_set(n, min_elements, max_elements, number=True, add_text=None,
                      file_name=None, path=None, include_solutions=False, include_seed=False):
    images = []
    if include_solutions:
        solution_images = []
    # make seeds before CircuitState starts controlling the RNG
    random.seed(None)
    seeds = [seeder() for _ in range(n)]
    for i in range(n):
        cs = CircuitState(min_elements=min_elements, max_elements=max_elements, seed=seeds[i])
        circuit = Circuit(circuitstate=cs)
        img = circuit.draw()
        texts = []
        if number:
            texts.append(f"Circuit number {i+1}")
        if add_text:
            texts.append(add_text)
        if include_seed:
            texts.append(f"seed: {seeds[i]} (CircuitPractice v. {version})")
        images.append(add_text_to_img(img, '\n'.join(texts)))
        if include_solutions:
            sol_img = circuit.draw(with_solution=True)
            sol_texts = texts
            if number:
                sol_texts[0] = f"Solution circuit number {i+1}"
            solution_images.append(add_text_to_img(sol_img, '\n'.join(sol_texts)))
    if file_name is None:
        return images, solution_images
    if include_solutions:
        images.append(solution_interpage)
        images.extend(solution_images)
    make_pdf(images, file_name=file_name, path=path)


if __name__ == '__main__':
    # logging.basicConfig(level=logging.DEBUG)
    # logging.basicConfig(level=logging.WARNING)
    logging.basicConfig(level=logging.ERROR)  # pick your poison
    easy = ["easy", {"Resistor": 1, "Diode": 1, "Zener": 0}, {"Resistor": 3, "Diode": 2, "Zener": 1}]
    medium = ["medium", {"Resistor": 2, "Diode": 1, "Zener": 0}, {"Resistor": 5, "Diode": 3, "Zener": 2}]
    hard = ["hard", {"Resistor": 3, "Diode": 2, "Zener": 1}, {"Resistor": 7, "Diode": 5, "Zener": 4}]
    for level_name, *params in (easy, medium, hard):
        images, solution_images = [], []
        text = f"Level: {level_name}.\n   " + \
               '\n   '.join(f"{params[0][comp]} <= {comp}s <= {params[1][comp]}"for comp in hard[-1].keys())
        this_img, this_sol_img = make_exercise_set(50, *params, number=True, add_text=text,
                                                   include_solutions=True, include_seed=True)
        images.extend(this_img)
        solution_images.extend(this_sol_img)
        make_pdf(images + [solution_interpage] + solution_images,
                 file_name=f"Exercises_{level_name}.pdf")