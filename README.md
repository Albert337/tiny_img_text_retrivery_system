Implementing text search and image search based on clip algorithm

- how to run
  
step 1:
```
  pip install -r requirements.txt

  python scripts/download_model.py  # download model

```
and the image data cna be download from the internet.

 step 2: 
 you should select an vector db to save the embedding,such as milvus or faiss.
 ```
 docker pull milvusdb/milvus:v2.4.14

###单独启动
docker run -d \
    --name milvus-standalone \
    --security-opt seccomp:unconfined \
    -e ETCD_USE_EMBED=true \
    -e ETCD_DATA_DIR=/var/lib/milvus/etcd \
    -e ETCD_CONFIG_PATH=/milvus/configs/embedEtcd.yaml \
    -e COMMON_STORAGETYPE=local \
    -v $(pwd)/volumes/milvus:/var/lib/milvus \
    -v $(pwd)/embedEtcd.yaml:/milvus/configs/embedEtcd.yaml \
    -v $(pwd)/user.yaml:/milvus/configs/user.yaml \
    -p 19530:19530 \
    -p 9091:9091 \
    -p 2379:2379 \
    --health-cmd="curl -f http://localhost:9091/healthz" \
    --health-interval=30s \
    --health-start-period=90s \
    --health-timeout=20s \
    --health-retries=3 \
    milvusdb/milvus:v2.4.14

  ###test connect
  python scripts/test_connect.py
  ```

step 3:
get the embedding of datasets and save them to vector db
```
  python scripts/insert_search.py
  python scripts/main.py
```

- the result is shown on like this:
![image](https://github.com/user-attachments/assets/bdcbdd69-e728-4a57-80dc-907d50237233)
