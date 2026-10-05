## Cloze probability, ratings, and computational predictability estimates
Scripts and data for _cloze probability responses_, _predictability ratings_ and _Transformer-based surprisal estimates_ for 205 sentences (1,726 words) from the UCL reading corpus.

A detailed description of the dataset can be found in [our paper](https://link.springer.com/article/10.3758/s13428-023-02261-8). If you use this resource, please cite it (see [How to cite](#how-to-cite)).

The resource we release is aligned with:
- :eyes: **Behavioral data**
  - Self-paced reading time ([Frank et al., 2013](https://link.springer.com/article/10.3758/s13428-012-0313-y))
  - Eye-tracking data ([Frank et al., 2013](https://link.springer.com/article/10.3758/s13428-012-0313-y))
    1. First fixation duration
    2. Gaze duration
    3. Right-bounded reading time
    4. Go-past reading time
- :electric_plug: **EEG data** ([Frank et al., 2015](https://www.sciencedirect.com/science/article/pii/S0093934X14001515))
  - N400; EPNP; PNP; P600; ELAN; LAN.

### Our dataset

Our dataset of cloze probability and predictability ratings is in the file `ratings_and_cloze.csv`; it is obtained from the item set `item-set.csv` from the UCL reading corpus [(Frank et al. 2013)](https://link.springer.com/article/10.3758/s13428-012-0313-y). This dataset is merged with the behavioral and neural measures described above in the dataframe `all_measures.csv`. The raw data (Prolific exports) can be found in the folders cp (cloze probability) and ratings. 

We also release the cloze distributions (i.e., not only the probability assigned to the target words, but to all the words that were produced in the cloze task). They can be found in the `cloze_distribution` folder, both in `.txt` and `.pkl` format.

:heavy_exclamation_mark: **Important note:**
If you use the neural and behavioral data, or the older probabilistic estimates (RNN, PSG, _N_-grams) please cite:
- For **EEG** data and older probabilistic estimates (RNN, PSG, _N_-grams): 
  - Frank, S. L., Otten, L. J., Galli, G., & Vigliocco, G. (2015). The ERP response to the amount of information conveyed by words in sentences. _Brain and language_, 140, 1-11.
- For **behavioral** data:
  - Frank, S. L., Fernandez Monsalve, I., Thompson, R. L., & Vigliocco, G. (2013). Reading time data for evaluating broad-coverage models of English sentence processing. _Behavior research methods_, 45, 1182-1190.

### The code

The code for our analyses is divided in four scripts:
- `preprocessing.py`, which performs data cleaning and aggregation of results.
- `merge_with_behavioural_data`, which merges our measurements with the neural and behavioural indexes of processing difficulty released by [Frank et al. (2013,](https://link.springer.com/article/10.3758/s13428-012-0313-y)[ 2015)](https://www.sciencedirect.com/science/article/pii/S0093934X14001515).
- `get_LM_surprisal.py`, which extracts surprisal values (negative log-probabilities) for the words in our dataset from Transformer-based language models released on the [HuggingFace Hub](https://huggingface.co/models).
  - Surprisal is defined as $s(w_i) = -\log p(w_i | w_1, w_2, \ldots w_{i-1}) $
- `plot.py`, which performs descriptive and inferential analyses and plots the results.

### Supplementary materials
In the folder `supplementary_materials` you can find the complete results of the analyses we reported in our paper in a more searchable csv format.

### :bookmark: How to cite

**If you use any part of this resource**, please cite our paper:

> de Varda, A. G., Marelli, M., & Amenta, S. (2024). [Cloze probability, predictability ratings, and computational estimates for 205 English sentences, aligned with existing EEG and reading time data.](https://link.springer.com/article/10.3758/s13428-023-02261-8) *Behavior Research Methods*, 56(5), 5190-5213.

**If you also use the data from Frank et al.**, please cite their papers too:

- **Reading time data** (self-paced reading, eye-tracking):
  > Frank, S. L., Fernandez Monsalve, I., Thompson, R. L., & Vigliocco, G. (2013). [Reading time data for evaluating broad-coverage models of English sentence processing.](https://link.springer.com/article/10.3758/s13428-012-0313-y) *Behavior Research Methods*, 45(4), 1182-1190.
- **EEG data** and the **older probabilistic estimates** (RNN, PSG, _N_-grams):
  > Frank, S. L., Otten, L. J., Galli, G., & Vigliocco, G. (2015). [The ERP response to the amount of information conveyed by words in sentences.](https://www.sciencedirect.com/science/article/pii/S0093934X14001515) *Brain and Language*, 140, 1-11.

<details>
<summary>BibTeX</summary>

```bibtex
@article{devarda2024cloze,
  title   = {Cloze probability, predictability ratings, and computational estimates for 205 English sentences, aligned with existing EEG and reading time data},
  author  = {de Varda, Andrea Gregor and Marelli, Marco and Amenta, Simona},
  journal = {Behavior Research Methods},
  volume  = {56},
  number  = {5},
  pages   = {5190--5213},
  year    = {2024},
  doi     = {10.3758/s13428-023-02261-8}
}

@article{frank2013reading,
  title   = {Reading time data for evaluating broad-coverage models of English sentence processing},
  author  = {Frank, Stefan L. and Fernandez Monsalve, Irene and Thompson, Robin L. and Vigliocco, Gabriella},
  journal = {Behavior Research Methods},
  volume  = {45},
  number  = {4},
  pages   = {1182--1190},
  year    = {2013},
  doi     = {10.3758/s13428-012-0313-y}
}

@article{frank2015erp,
  title   = {The {ERP} response to the amount of information conveyed by words in sentences},
  author  = {Frank, Stefan L. and Otten, Leun J. and Galli, Giulia and Vigliocco, Gabriella},
  journal = {Brain and Language},
  volume  = {140},
  pages   = {1--11},
  year    = {2015},
  doi     = {10.1016/j.bandl.2014.10.006}
}
```
</details>

### License
- **Our data** (cloze responses, predictability ratings, cloze distributions, surprisal estimates): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). You can use and adapt it freely, as long as you cite our paper (see [How to cite](#bookmark-how-to-cite)).
- **Code**: [MIT License](LICENSE).
- **Not covered by this license:** the sentences (`item-set.csv`) and the reading time and EEG measures in `all_measures.csv`. These come from Frank et al. (2013, 2015) and are redistributed here with permission from Stefan Frank. They remain under their original terms, so please cite the original papers (see [How to cite](#bookmark-how-to-cite)).

### Contact :envelope:
If you have any troubles with the resource, please do not hesitate and contact me at `devar_ag` 
