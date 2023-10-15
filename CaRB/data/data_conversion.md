# Data Conversion Guideline and Scripts

## Overview
The purpose of data conversion pipeline for CaRB, is to create a set of training data for OpenIE, that is compatible with the required triple format for Entailment Graph Construction.

## Conversion Guideline
1. annotate prepositions for each object when applicable;
2. do not cut away prepositional modifiers from arguments, only cut away prepositional modifiers from predicates;
3. extract embedded triples in ``said'' / ``claimed'' structures;
4. move modals and negations to "auxilliary";
5. keep the original order of elements;

## Scripts

- Reformat CaRB data to prepare for annotation: ` python data_conversion.py --task rfmt_anno --subset [dev/test] `
- Do Conversion with GPT-4: ` python data_conversion.py --task gpt_annotate --subset [dev/test] `
- Randomly (`seed = 42`) split the original test set into test1 / test2: `cd ./CaRBent_gold ; python tmp.py `

Resulting files are saved in `./CaRBent_gold/` directory.


