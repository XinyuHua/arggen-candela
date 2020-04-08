"""Rule-based filtering over chunking results"""
import json

STOPWORDS = [w.lower() for w in open('dat/stopwords.txt')]
WIKI_TITLE = [] # NEED TO BE SUBSTITUTE WITH REAL DATA

def filter_kp(candidates, topic_signatures):
    """Rule-based keyphrase filtering:
        1. if all words are stopwords, remove
        2. if none of the words are alphabetical, remove
        3. if more than 10 tokens, remove
        4. if contains any topic signature words, keep
        5. if contains Wikipedia title, keep
    """

    result = []
    for ph in candidates:
        if len(ph.split()) > 10: continue

        contains_nonstop = False
        contains_alphbet = False
        contains_ts = False

        for token in ph.split():
            token = token.lower()
            token_no_hyphen = token.replace('-', '')
            if str.isalpha(token_no_hyphen):
                contains_alphbet = True
            if not token in STOPWORDS:
                contains_nonstop = True
            if token in topic_signatures and not token in STOPWORDS:
                contains_ts = True

        if not contains_nonstop: continue
        if not contains_alphbet: continue

        contains_wiki_title = False
        ph_lower = ph.lower()
        if ph_lower in WIKI_TITLE:
            contains_wiki_title = True

        elif ph_lower.replace('-', '') in WIKI_TITLE:
            contains_wiki_title = True

        elif ph_lower.replace('the', '').strip() in WIKI_TITLE:
            contains_wiki_title = True

        elif ph_lower.replace('a', '').strip() in WIKI_TITLE:
            contains_wiki_title = True

        def compare_with_chosen(result, ph):
            """If current candidate has overlap with any chosen one,
            take the longer one
            """
            already_chosen = False
            for chosen_ph in result:
                if ph in chosen_ph:
                    chosen_ph_first_tok = chosen_ph.split()[0]
                    if chosen_ph_first_tok in ['believe', 'think']:
                        result.remove(chosen_ph)
                    else:
                        already_chosen = True


                if chosen_ph in ph:
                    #print(f'chosen phrase {chosen_ph} covered by current phrase {ph}')
                    ph_first_tok = ph.split()[0]
                    if not ph_first_tok in ['think', 'believe']:
                        result.remove(chosen_ph)

                    else:
                        already_chosen = True

            if not already_chosen:
                result.append(ph)
            return result

        if contains_ts:
            compare_with_chosen(result, ph)
        elif contains_wiki_title:
            compare_with_chosen(result, ph)
    return result

def demo():
    topic_signatures = [json.loads(ln) for ln in open('dat/sample_topic_signatures.jsonl')]
    phrases = [json.loads(ln) for ln in open('dat/sample_phrases.jsonl')]
    fout = open('dat/sample_keyphrases.jsonl', 'w')

    for ix, (ts, ph) in enumerate(zip(topic_signatures, phrases)):
        candidates = []
        for sent in ph['phrase_list']:
            for np in sent['np_list'] + sent['vp_list']:
                candidates.append(np)

        ret = filter_kp(candidates, ts['topic_signature'])
        ret_obj = dict(
                keyphrase=ret,
                index=ix)
        fout.write(json.dumps(ret_obj) + '\n')
        print(f'sample {ix}: {len(ret)} phrases retained from {len(candidates)}')

    fout.close()

if __name__=='__main__':
    demo()

