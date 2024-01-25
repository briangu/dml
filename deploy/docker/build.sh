cd $1
docker buildx build -t $2 .
cd ..
