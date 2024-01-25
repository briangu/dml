cd $1
docker buildx build --no-cache -t $2 .
cd ..
