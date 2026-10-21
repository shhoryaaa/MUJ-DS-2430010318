# Assignment 2 — Dataset and Exploratory Data Analysis

## Dataset summary
The exported notebook reports 5,606 matched images and 4,230 unique patients
from the NIH ChestX-ray14 sample. It includes 14 disease labels.

## EDA questions
1. How many image records are available after matching paths?
2. How many unique patients exist?
3. What is the frequency of each label?
4. How common is the No Finding category?
5. Are images available in different view positions?
6. Are there duplicate filenames or unreadable images?
7. How many positive examples of each label appear in each split?

## Multi-label counting
A row containing `Effusion|Mass` contributes one positive example to Effusion
and one to Mass. Consequently, disease counts need not sum to the dataset size.

## Important caution
A high count for a label does not imply that a model detects it accurately.
Counts describe the dataset; predictive performance must be measured separately.
