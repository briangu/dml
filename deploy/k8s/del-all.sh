kubectl delete -f dml-discovery.yml

# find all -job.yml files and delete them from k8s
for f in $(find . -name "*-job.yml"); do
    kubectl delete -f $f
done

