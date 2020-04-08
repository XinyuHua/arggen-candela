"""Run pycorenlp over data to obtain annotations"""
import json
from pycorenlp import StanfordCoreNLP

def run_sample():
    data = [ln.strip() for ln in open('dat/sample.txt')]
    nlp = StanfordCoreNLP('http://localhost:9000')

    fout = open('dat/sample_annotated.jsonl', 'w')
    for ln in data:
        output = nlp.annotate(ln, properties={
                'annotators': 'tokenize,ssplit,pos',
                'outputFormat': 'json'})
        fout.write(json.dumps(output) + '\n')
    fout.close()


if __name__=='__main__':
    run_sample()
