@echo off

python -m cProfile -s tottime main.py >> profile.txt
