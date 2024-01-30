cd $1
#docker buildx build --platform linux/amd64,linux/arm64 --no-cache -t $2 --push .
docker buildx build --platform linux/amd64,linux/arm64 -t $2 --load .
docker push $2
cd ..
