cd $1
docker buildx build --platform linux/amd64,linux/arm64 --no-cache -t $2 --push .
cd ..
