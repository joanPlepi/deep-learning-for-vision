#!/bin/bash  
a=1
b="img_"

# to avoid renaming
for i in *.jpg *.png; do
  new=$(printf "%s.jpg" "$b$a")
  mv -- "$i" "$new"
  let a=a+1
done

a=1
for i in *.jpg; do
  new=$(printf "%d.jpg" "$a")
  mv -- "$i" "$new"
  let a=a+1
done