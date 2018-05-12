from __init__ import preprocess_corpus
import argparse

parser = argparse.ArgumentParser(description="Preprocess corpus")
parser.add_argument('corpus_in', type=str, help='input corpus')        
parser.add_argument('corpus_out', type=str, help='output (preprocessed) corpus')            
parser.add_argument('-max_sent', type=int, help='max number of sentences to be proces')
parser.add_argument('-sep_emoji', action="store_true", default=False, help='separate emojis')

args = parser.parse_args()
if args.max_sent:
	preprocess_corpus(args.corpus_in, args.corpus_out, sep_emoji=args.sep_emoji, max_sent=args.max_sent)
else:
	preprocess_corpus(args.corpus_in, args.corpus_out, sep_emoji=args.sep_emoji)
