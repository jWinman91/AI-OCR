sudo apt install python3-pip
sudo apt-get update && apt-get install -y ninja-build
sudo apt-get install poppler-utils
pip install -r requirements.txt
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 --verbose
