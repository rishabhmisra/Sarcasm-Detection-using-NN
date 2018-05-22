step=$1
TRAIN="DATA/txt/bamman_clean.txt"
WORD_EMBS="DATA/embeddings/filtered_embs.txt"
USER_EMBS="DATA/embeddings/usr2vec.txt"
THEANO_FLAGS='floatX=float32' python code/crossfold_run.py DATA/pkl/cross.pkl ${TRAIN} \
							-idField 0 -userField 1 -tagField 2 -textField 3 \
							-vectors ${WORD_EMBS} \
							-user_vectors ${USER_EMBS} \
							-folds DATA/folds/fold_${step}.csv  \
							-context 1 

