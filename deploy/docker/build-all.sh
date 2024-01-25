./build.sh dml-discovery eismcc/dml-discovery:latest
./build.sh dml-mnist-cpu eismcc/dml-mnist-cpu:latest
docker push eismcc/dml-discovery:latest
docker push eismcc/dml-mnist-cpu:latest
