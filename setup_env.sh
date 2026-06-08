#!/bin/bash
echo "--- Python Sanal Ortam Kuruluyor ---"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "--- Kurulum Tamamlandı ---"
echo "Ortamı aktifleştirmek için: source .venv/bin/activate"
