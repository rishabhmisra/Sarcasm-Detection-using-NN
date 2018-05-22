if [[ "$#" == "2" ]]; then 
	word_embeddings=$1
	user_embeddings=$2
	mkdir DATA
	mkdir DATA/embeddings
	mkdir DATA/pkl
	mkdir DATA/txt
	rm DATA/embeddings/words.txt
	rm DATA/embeddings/usr2vec.txt
	echo "word embeddings: " ${word_embeddings}
	echo "user embeddings: " ${user_embeddings} 
	ln -s ${word_embeddings} DATA/embeddings/words.txt
	ln -s ${user_embeddings} DATA/embeddings/usr2vec.txt
	python code/extract.py
  else
	echo "missing parameters. Please run as: ./prepare.sh [path_to_word_embeddings] [path_to_user_embeddings]"
fi

