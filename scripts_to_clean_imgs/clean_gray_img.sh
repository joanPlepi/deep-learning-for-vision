#!/bin/bash  
a=1
for i in *.jpg *.png; do
  if  file $i | grep -i 'frames 1'; then
  	rm $i
  fi
done