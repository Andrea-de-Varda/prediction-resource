## Cloze probability, ratings, and computational predictability estimates
Scripts and data relative to the _cloze probability responses_, _predictability ratings_ and _Transformer-based surprisal estimates_ for 205 sentences (1,726 words).

The resource we release is aligned with:
- **Behavioral data**
  - Self-paced reading time ([Frank et al., 2013](https://link.springer.com/article/10.3758/s13428-012-0313-y))
  - Eye-tracking data ([Frank et al., 2013](https://link.springer.com/article/10.3758/s13428-012-0313-y))
    1. First fixation duration
    2. Gaze duration
    3. Right-bounded reading time
    4. Go-past reading time
- **EEG data** ([Frank et al., 2015](https://www.sciencedirect.com/science/article/pii/S0093934X14001515))
  - N400; EPNP; PNP; P600; ELAN; LAN.

### Our dataset

Our dataset of cloze probability and predictability ratings is in the file `ratings_and_cloze.csv`; it is obtained from the item set `item-set.csv` from the UCL reading corpus [(Frank et al. 2013)](https://link.springer.com/article/10.3758/s13428-012-0313-y). The raw data (Prolific exports) can be found in the folders cp (cloze probability) and ratings. 

### The code

The code for our analyses is divided in four scripts:
- `preprocessing.py`, which performs data cleaning and aggregation of results.
- `merge_with_behavioural_data`, which merges our measurements with the neural and behavioural indexes of processing difficulty released by [Frank et al. (2013,](https://link.springer.com/article/10.3758/s13428-012-0313-y)[ 2015)](https://www.sciencedirect.com/science/article/pii/S0093934X14001515).
- `get_LM_surprisal.py`, which extracts surprisal values (negative log-probabilities) for the words in our dataset from Transformer-based language models released on the [HuggingFace Hub](https://huggingface.co/models).
  - Surprisal is defined as $s(w_i) = -\log p(w_i | w_1, w_2, \ldots w_{i-1}) $
- `plot.py`, which performs descriptive and inferential analyses and plots the results.
