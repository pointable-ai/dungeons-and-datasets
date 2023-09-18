#!/bin/bash

mkdir -p model
wget https://huggingface.co/TheBloke/Nous-Hermes-Llama2-GGUF/resolve/main/nous-hermes-llama2-13b.Q4_K_S.gguf -O model/nous-hermes-llama2-13b-q4.gguf
# wget https://huggingface.co/TheBloke/Nous-Hermes-Llama-2-7B-GGUF/resolve/main/nous-hermes-llama-2-7b.Q4_K_M.gguf -O model/nous-hermes-llama-2-7b-q4.gguf