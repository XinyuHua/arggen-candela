""" extract NP and VP from processed data """
import json
from nltk.chunk import regexp
from nltk import Tree
import sys

def chunk_tagged_sents(tagged_sent):
    grammar = r"""
        NP: {<DT|PP\$>?<JJ|JJR>*<NN.*|CD|JJ>+}
        PP: {<IN><NP>}
        VP: {<MD>?<VB.*><NP|PP>}
        CLAUSE: {<NP><VP>}
        """
    chunker = regexp.RegexpParser(grammar, loop=2)
    chunked_sent = chunker.parse(tagged_sent)
    return chunked_sent

def get_chunks(chunked_sent):
    np_chunks = []
    vp_chunks = []
    np_raw_chunks = []
    vp_raw_chunks = []
    for subtree in chunked_sent.subtrees():
        if subtree.label() == "NP":
            np_raw_chunks.append(subtree.leaves())
        elif subtree.label() == "VP":
            vp_raw_chunks.append(subtree.leaves())
    for np_raw_chunk in np_raw_chunks:
        chunk = []
        for word_tag in np_raw_chunk:
            chunk.append(word_tag[0])
        np_chunks.append(' '.join(chunk))
    for vp_raw_chunk in vp_raw_chunks:
        chunk = []
        for word_tag in vp_raw_chunk:
            chunk.append(word_tag[0])
        vp_chunks.append(' '.join(chunk))
    return np_chunks, vp_chunks

def extract_kp_from_processed():
    data = []
    fout = open("dat/sample_phrases.jsonl", 'w')
    data = [json.loads(ln) for ln in open("dat/sample_annotated.jsonl")]

    for ix, cur_obj in enumerate(data):
        phrase_list = []

        for sent in cur_obj['sentences']:
            cur_pos = [(tok["originalText"], tok["pos"]) for tok in sent["tokens"]]
            cur_words = [tok["originalText"] for tok in sent["tokens"]]
            cur_np_lst, cur_vp_lst = get_chunks(chunk_tagged_sents(cur_pos))
            phrase_list.append({"np_list": cur_np_lst,
                               "vp_list": cur_vp_lst})

        ret_obj = dict(
                phrase_list=phrase_list,
                index=ix)
        fout.write(json.dumps(ret_obj) + '\n')
    fout.close()
    return



if __name__=='__main__':
    extract_kp_from_processed()
