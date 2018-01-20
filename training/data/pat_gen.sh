#!/usr/bin/env bash
echo 'generate hyphenation pattern'
pypatgen patterns new patterns.txt
pypatgen patterns train -r 1,3 -s 1:5:1 --commit
pypatgen patterns train -r 1,4 -s 1:5:1 --commit
pypatgen patterns train -r 1,5 -s 1:10:1 --commit
pypatgen patterns train -r 1,6 -s 1:10:1 --commit
echo 'done'
