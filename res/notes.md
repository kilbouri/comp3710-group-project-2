# Development Notes

## Input Data

**Files**

All data files should be located in `/data/`. Additionally, there are two files that must be present:

- an attribute mapping file (`attributes.dat`, by default), which contains the names of each attribute. The default name for the classification attribute is `classification`.
- a data file containing the input examples.

**The input data should meet the following criteria:**

- each line of the data file is a single example
- each line consists only of comma-separated values
- there are exactly as many elements in each example of the input data as there are in the attribute names specified in the attribute mapping file
- a 'yes' classification ('edible' in the mushroom dataset) is indicated by the letter `e`, which is short for edible, as this was developed for classifying edible vs. non-edible.
