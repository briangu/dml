kubectl get node -o json | jq '.items[].metadata | {hostname: .name, gpu_product: .labels."nvidia.com/gpu.product", gpu_memory: .labels."nvidia.com/gpu.memory"} '
