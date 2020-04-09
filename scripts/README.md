## Sample scripts on pre-processing

------------------------

### Topic Signature

We conduct log-likelihood ratio test to obtain topic signature words. Here we
use 1000 ChangeMyView thread OP as an example. Suppose they are first lemmatized
and stored in `dat/cmv_op_lemma.jsonl`, then the following command will calculate
log-likelihood ratio test for each document (OP). For each document, we consider
all others to be the background corpus. The results will be saved to `dat/cmv_op_llr.jsonl`.

```shell script
python calculate_topic_signatures.py
```

To obtain the topic signature words, simply load the log-likelihood ratio results
and keep the ones with more than `10.83` scores (corresponding to confidence level 0.001).

For more details please check this paper:

```
@inproceedings{lin-hovy-2000-automated,
title = "The Automated Acquisition of Topic Signatures for Text Summarization",
author = "Lin, Chin-Yew  and
  Hovy, Eduard",
booktitle = "{COLING} 2000 Volume 1: The 18th International Conference on Computational Linguistics",
year = "2000",
url = "https://www.aclweb.org/anthology/C00-1072",
}
```

###  Keyphrase Identification

Step 1: process with Stanford CoreNLP (using pycorenlp wrapper):

```shell script
python run_core_nlp.py
```

The result will be saved as `dat/sample_annotated.jsonl`.


Step 2: extract phrases from CoreNLP's output:

```shell script
python extract_phrase.py
```

The result will be saved as `dat/sample_phrases.jsonl`.

Step 3: filtering, to retrain keyphrases only.

Note: we omit loading WordNet lexicon and wikipedia titles for this demo. We 
also assume the topic signature words to be pre-computed and stored at `dat/sample_topic_signatures.jsonl`

```shell script
python filter_keyphrase.py
```

Result will be saved as `dat/sample_keyphrases.jsonl`



