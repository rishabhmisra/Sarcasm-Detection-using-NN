MODEL="DATA/pkl/sarcasm_cnn"
TRAINING_DATA="DATA/txt/bamman_clean.txt"
WORD_EMB="DATA/embeddings/filtered_embs.txt"
USER_EMB="DATA/embeddings/usr2vec.txt "
python code/train_cnn.py ${MODEL} ${TRAIN_DATA} -train \
                                                -userField 1 -tagField 2 -textField 3 \
                                                -vectors ${WORD_EMB} \
                                                -user_vectors ${USER_EMB} \
                                                -epochs 10