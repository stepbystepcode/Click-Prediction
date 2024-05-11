#!/usr/bin/zsh
source ~/.zshrc
mamba activate lj-nlp
export no_proxy="localhost,127.0.0.1"
python3 ui.py
