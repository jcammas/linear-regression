#!/bin/bash
python3 -m pip install --user --upgrade pip
if command -v apt-get &> /dev/null; then
    sudo apt-get install python3-venv
    sudo apt install python3-tk
fi
python3 -m venv venv
source venv/bin/activate
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "success"
    pip install -r install.txt
else
    echo "error"
fi