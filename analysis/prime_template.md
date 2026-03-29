# STARK_PRIME Templates

| Id | Template | Hops | Entities (#) | Entities | Relations (#) | Relations |
|----|----------|------|--------------|----------|---------------|-----------|
| 1 | (effect/phenotype → [phenotype absent] → disease ← [!indication] ← drug) | 1 | 3 | effect/phenotype, disease, drug | 2 | phenotype_absent, contraindication |
| 2 | (drug → [contraindication] → disease ← [associated with] ← gene/protein) | 1 | 3 | gene/protein, disease, drug | 2 | associated_with, contraindication |
| 3 | (anatomy → [expression present] → gene/protein ← [expression absent] ← anatomy) | 1 | 2 | gene/protein, anatomy | 2 | expression_present, expression_absent |
| 4 | (anatomy → [expression absent] → gene/protein ← [expression absent] ← anatomy) | 1 | 2 | anatomy, gene/protein | 2 | expression_present, expression_absent |
| 5 | (drug → [carrier] → gene/protein ← [carrier] ← drug) | 1 | 2 | drug, gene/protein | 1 | carrier |
| 6 | (anatomy → [expression present] → gene/protein → [target] → drug) | 2 | 3 | drug, gene/protein, anatomy | 2 | expression_present, target |
| 7 | (drug → [side effect] → effect/phenotype → [side effect] → drug) | 2 | 2 | drug, effect/phenotype | 1 | side_effect |
| 8 | (drug → [carrier] → gene/protein → [carrier] → drug) | 2 | 2 | drug, gene/protein | 1 | carrier |
| 9 | (anatomy → [expression present] → gene/protein → enzyme → drug) | 2 | 3 | anatomy, gene/protein, drug | 2 | expression_present, enzyme |
| 10 | (cellular_component → [interacts with] → gene/protein → [carrier] → drug) | 2 | 3 | cellular_component, gene/protein, drug | 2 | interacts_with, carrier |
| 11 | (molecular_function → [interacts with] → gene/protein → [target] → drug) | 2 | 3 | molecular_function, gene/protein, drug | 2 | interacts_with, target |
| 12 | (effect/phenotype → [side effect] → drug → [synergistic interaction] → drug) | 2 | 2 | effect/phenotype, drug | 2 | side_effect, synergistic_interaction |
| 13 | (disease → [indication] → drug → [contraindication] → disease) | 2 | 2 | disease, drug | 2 | indication, contraindication |
| 14 | (disease → [parent-child] → disease → [phenotype present] → effect/phenotype) | 2 | 2 | disease, effect/phenotype | 2 | parent_child, phenotype_present |
| 15 | (gene/protein → [transporter] → drug → [side effect] → effect/phenotype) | 2 | 3 | gene/protein, drug, effect/phenotype | 2 | transporter, side_effect |
| 16 | (drug → [transporter] → gene/protein → [interacts with] → exposure) | 2 | 3 | drug, gene/protein, exposure | 2 | transporter, interacts_with |
| 17 | (pathway → [interacts with] → gene/protein → [ppi] → gene/protein) | 2 | 2 | pathway, gene/protein | 2 | interacts_with, ppi |
| 18 | (drug → [synergistic interaction] → drug → [transporter] → gene/protein) | 2 | 2 | drug, gene/protein | 2 | synergistic_interaction, transporter |
| 19 | (biological_process → [interacts with] → gene/protein → [interacts with] → biological_process) | 2 | 2 | biological_process, gene/protein | 1 | interacts_with |
| 20 | (effect/phenotype → [associated with] → gene/protein → [interacts with] → biological_process) | 2 | 3 | effect/phenotype, gene/protein, biological_process | 2 | associated_with, interacts_with |
| 21 | (drug → [transporter] → gene/protein → [expression present] → anatomy) | 2 | 3 | drug, gene/protein, anatomy | 2 | transporter, expression_present |
| 22 | (drug → [target] → gene/protein → [interacts with] → cellular_component) | 2 | 3 | drug, gene/protein, cellular_component | 2 | target, interacts_with |
| 23 | (biological_process → [interacts with] → gene/protein → [expression absent] → anatomy) | 2 | 3 | biological_process, gene/protein, anatomy | 2 | interacts_with, expression_absent |
| 24 | (effect/phenotype → [associated with] → gene/protein → [expression absent] → anatomy) | 2 | 3 | effect/phenotype, gene/protein, anatomy | 2 | associated_with, expression_absent |
| 25 | (drug → [indication] → disease → [indication] → drug) & (drug → [synergistic interaction] → drug) | 3 | 2 | drug, disease | 2 | indication, synergistic_interaction |
| 26 | (pathway → [interacts with] → gene/protein → [interacts with] → pathway) & (pathway → [parent-child] → pathway) | 3 | 2 | pathway, gene/protein | 2 | interacts_with, parent_child |
| 27 | (gene/protein → [associated with] → disease → [associated with] → gene/protein) & (gene/protein → [ppi] → gene/protein) | 3 | 2 | gene/protein, disease | 2 | associated_with, ppi |
| 28 | (gene/protein → [associated with] → effect/phenotype → [associated with] → gene/protein) & (gene/protein → [ppi] → gene/protein) | 3 | 2 | gene/protein, effect/phenotype | 2 | associated_with, ppi |