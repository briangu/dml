cd $1
docker buildx build --platform linux/amd64 -t $2 --push .
cd ..
