kubectl apply -f dml-discovery.yml
sleep 20
#kubectl apply -f dml-mnist-cpu-job.yml
python3 run-job.py dml-mnist-cpu-job.yml
kubectl get pods -n dml
