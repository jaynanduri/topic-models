# LDA and NMF
***
Topic Models (K=10, 20, 50) for 20NG and DUC 2001 datasets by running LDA and NMF methods. 
Printing out for each topic the top 20 words (with probabilities).

# Extractive summarization
***
Implemented the KL-Sum summarization method for both DUC 2001 dataset. 
Reference - [paper](extractive_summarization.pdf)

- KL_summary based on words_PD; PD is a distribution proportional to counts of words in document
- LDA_summary based on LDA topics_PD on obtained in PB2. The only difference is that PD, while still a distribution over words, is computed using topic modeling
- For DUC dataset evaluated KL_summaries and LDA_summaries against human gold summaries with ROUGE. [ROGUE PERL Package](https://github.com/andersjo/pyrouge/tree/master/tools/ROUGE-1.5.5) 
