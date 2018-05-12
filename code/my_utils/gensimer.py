import argparse
from collections import Counter
from gensim.models.word2vec import Word2Vec
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from ipdb import set_trace
import pprint 
from __init__ import word_2_idx

class Word2VecReader(object):
	def __init__(self, datasets, max_sent=None):
		self.datasets = datasets
		self.max_sent = max_sent if max_sent else float('inf')
		if self.max_sent < float('inf'):
			print "[max_sentences: %d]" % self.max_sent
	def __iter__(self):		
		for dataset in self.datasets:
			print dataset
			lines=0	
			with open(dataset) as fid:
				for l in fid:		
					lines+=1
					if lines>self.max_sent: break
					yield l.decode("utf-8").split()

class Doc2VecReader(object):
	"""
		IMPORTANT: this reader assumes that each line in the file has the structure <paragraph_id>\t<sentence>
		All sentences with the same <paragrah_id> will be considered as one paragraph
	"""
	def __init__(self, datasets, max_sent=None):
		self.datasets = datasets
		self.max_sent = max_sent if max_sent else float('inf')
		if self.max_sent < float('inf'):
			print "[max_sentences: %d]" % self.max_sent
	def __iter__(self):		
		for dataset in self.datasets:
			print dataset			
			with open(dataset) as fid:				
				for i, l in enumerate(fid):						
					if i>self.max_sent: break	
					txt = l.decode("utf-8").split()
					yield LabeledSentence(words=txt,tags=[i])

class LDAReader(object):
	def __init__(self, datasets, max_sent=None):
		"""
			datasets: datasets
			max_sent: maximum number of sentences to be read in each dataset			
		"""
		self.datasets = datasets
		self.max_sent = max_sent if max_sent else float('inf') 
		if self.max_sent < float('inf'):
			print "[max_sentences: %d]" % self.max_sent
		self._compute_vocabulary()

	def _compute_vocabulary(self):
		ct = Counter()
		for dataset in self.datasets:
			print dataset
			with open(dataset) as fid:
				for i,l in enumerate(fid):
					if i > self.max_sent: break
					ct.update(l.decode("utf-8").split())		
		self.wrd2idx = {w:i for i,w in enumerate(ct.keys())}
		self.idx2wrd = {i:w for w,i in self.wrd2idx.items()}
	
	def features(self, doc):
		ct = Counter(doc.decode("utf-8").split())
		return [(self.wrd2idx[w],c) for w,c in ct.items() if w in self.wrd2idx]

	def __iter__(self):		
		for dataset in self.datasets:
			print dataset			
			with open(dataset) as fid:
				for i,l in enumerate(fid):					
					if i>self.max_sent: break
					yield self.features(l)

def train_lda(args):
	print "[LDA > n_topics: %d ]" % args.dim	
	lda_reader = LDAReader(args.ds, max_sent=args.max_sent)		
	ldazito = LdaMulticore(lda_reader, id2word=lda_reader.idx2wrd,
									   num_topics=args.dim, 
									   workers=args.workers)
	ldazito.save(args.out)	

def train_skipgram(args):
	if args.negative_samples > 0:		
		print "[SKip-Gram > negative_samples: %d | min_count: %d | dim: %d | epochs: %d]" % (args.negative_samples, args.min_count, args.dim, args.epochs)
		w2v = Word2Vec(sentences=Word2VecReader(args.ds,args.max_sent), size=args.dim, 
			           workers=args.workers, min_count=args.min_count, sg=1, hs=0, 
			           negative=args.negative_samples, iter=args.epochs)		
	else:		
		print "[SKip-Gram (Hierachical Softmax) > min_count: %d | dim: %d | epochs: %d]" % (args.min_count, args.dim, args.epochs)
		w2v = Word2Vec(sentences=Word2VecReader(args.ds,args.max_sent), size=args.dim, 
			           workers=args.workers, min_count=args.min_count, sg=1, 
			           hs=1,iter=args.epochs)		
	w2v.train(Word2VecReader(args.ds,args.max_sent))
	w2v.save(args.out)
	w2v.save_word2vec_format(args.out+".txt")
	print "Done"	

def train_doc2vec(args):
	d2v_reader = Doc2VecReader(args.ds,args.max_sent)
	if args.negative_samples > 0:		
		print "[Doc2Vec > negative_samples: %d | min_count: %d | dim: %d | epochs: %d]" % (args.negative_samples, args.min_count, args.dim, args.epochs)		
		d2v = Doc2Vec(documents=d2v_reader, size=args.dim, 
			           workers=args.workers, min_count=args.min_count, hs=0, 
			           negative=args.negative_samples, iter=args.epochs)		
	else:		
		print "[Doc2Vec (Hierachical Softmax) > min_count: %d | dim: %d | epochs: %d]" % (args.min_count, args.dim, args.epochs)
		d2v = Doc2Vec(documents=d2v_reader, size=args.dim, 
			           workers=args.workers, min_count=args.min_count, 
			           hs=1,iter=args.epochs)		
	d2v_reader = Doc2VecReader(args.ds,args.max_sent)
	d2v.train(d2v_reader)		
	d2v.save(args.out)	
	print "Done"	

def get_parser():
	parser = argparse.ArgumentParser(description="Induce Text Representations with Gensim")
	parser.add_argument('-ds',  type=str, required=True, nargs='+', help='datasets')        
	parser.add_argument('-out', type=str, required=True, help='path to store the embeddings')
	parser.add_argument('-dim', type=int, required=True, help='size of embeddings or number of topics')
	parser.add_argument('-model',    choices=['w2v','doc2vec','lda'], required=True, help='model')
	parser.add_argument('-epochs',   type=int, default=5, help='number of epochs')
	parser.add_argument('-workers',  type=int, default=4, help='number of workers')
	parser.add_argument('-max_sent', type=int, help='set max number of sentences to be read (per file)')
	parser.add_argument('-min_count',type=int, default=10, help='words ocurring less than ''min_count'' times are discarded')
	parser.add_argument('-negative_samples', type=int, default=10, help='number of negative samples for Skip-Gram training. If set to 0 then Hierarchical Softmax will be used')
	return parser


if __name__ == "__main__":	
	cmdline_parser = get_parser()
	args = cmdline_parser.parse_args() 			
	print "** Induce Text Representations with Gensim **"
	print "[input > %s | max_sent: %s | workers: %d | output@%s]\n" % (repr(args.ds), repr(args.max_sent), args.workers,  args.out)	

	if args.model =="lda":
		train_lda(args)
	elif args.model == "doc2vec":
		train_doc2vec(args)
	elif args.model == "w2v":
		train_skipgram(args)
	else:		
		raise NotImplementedError, "unknown model: %s" % args.model
