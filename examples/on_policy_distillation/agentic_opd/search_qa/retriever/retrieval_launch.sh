#!/bin/bash
# Vendored from Search-R1 (https://github.com/PeterGriffinJin/Search-R1).
#
# Starts the E5 dense faiss retrieval service on 0.0.0.0:8000. Point $file_path
# at the dir holding the downloaded index + corpus (see searchr1_download.py /
# README.md); run ONCE, out of band, before training.

file_path=/the/path/you/save/corpus
index_file=$file_path/e5_Flat.index
corpus_file=$file_path/wiki-18.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

python "$(dirname "$0")/retrieval_server.py" --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path \
                                            --faiss_gpu
