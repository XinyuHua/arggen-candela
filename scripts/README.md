## Sample scripts on pre-processing

------------------------

The following scripts process each input plain text document into phrases.

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
