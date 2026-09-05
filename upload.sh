#!/bin/bash

for file in *; do
  [ -f "$file" ] || continue
  echo "Uploading $file"
  colab upload ./$file /content/$file
done
