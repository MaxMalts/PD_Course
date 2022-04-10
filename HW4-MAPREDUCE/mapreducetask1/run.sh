#! /usr/bin/env bash

OUT_DIR="result"

ant 1>/dev/null
hdfs dfs -rm -r -skipTrash ${OUT_DIR} 1>/dev/null
hadoop jar jar/IdShuffle.jar idshuffle.IdShuffle /data/ids ${OUT_DIR}

hdfs dfs -cat ${OUT_DIR}/part-r-00000 | python3 outputer.py 50