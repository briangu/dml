kubectl apply -f dml-discovery.yml
sleep 20
./run-job.sh $1
