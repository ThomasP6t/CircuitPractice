CircuitPractice
================

This is a program that generates 'nice' electronics circuits to solve as exercises.

When generating random exercises, it is rare that all values of currents, voltages and element parameters are 'nice',
in the sense that they are all round numbers (or up to one decimal place) and of similar order of magnitude. This program
nudges everything towards having rounder values.

After generating the exercise, it can be displayed on screen or written to disk, with or without the solution (i.e. all
current and voltage values).

The minimum and maximum of each type of component can be given, which controls the difficulty of the exercises generated.

Currently implements resistors, diodes, and zener diodes

To use:
---------
Download the files and run `main.py`. This will generate three sets of exercises: easy, medium and hard. If necessary,
the `main.py` file can be edited to tweak the paramters
(Accepting command line arguments is on the TODO list)
