#!/bin/bash

# Định nghĩa đường dẫn
INPUT_DIR="/user/hadoop/lung_cancer_data/lung_cancer_clean.csv"
OUTPUT_DIR="/user/hadoop/lung_cancer_wordcount"

# Xóa thư mục đầu ra nếu đã tồn tại
hdfs dfs -rm -r $OUTPUT_DIR

# Chạy Hadoop Streaming
hadoop jar $HADOOP_HOME/share/hadoop/tools/lib/hadoop-streaming-*.jar \
    -input $INPUT_DIR \
    -output $OUTPUT_DIR \
    -mapper "python3 mapper.py" \
    -reducer "python3 reducer.py" \
    -file mapper.py \
    -file reducer.py

echo "WordCount đã hoàn thành. Kết quả lưu tại: $OUTPUT_DIR"
