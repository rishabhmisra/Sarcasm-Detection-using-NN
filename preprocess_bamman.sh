EMB_PATH_IN="DATA/embeddings/words.txt"
EMB_PATH_OUT="DATA/embeddings/filtered_embs.txt"
TXT_IN="DATA/txt/bamman_redux.txt"
TXT_OUT="DATA/txt/bamman_clean.txt"
python aux/preprocess_bamman.py -input   ${TXT_IN} \
                                -out_txt ${TXT_OUT} \
                                -word_vectors ${EMB_PATH_IN} \
                                -out_vectors  ${EMB_PATH_OUT} \