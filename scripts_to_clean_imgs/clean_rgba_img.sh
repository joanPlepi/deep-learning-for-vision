#!/bin/bash  
for i in *.jpg *.png; do
  if file $i | grep -i -e 'RGBA' -e 'gray+alpha'; then
  	rm $i
  fi
done