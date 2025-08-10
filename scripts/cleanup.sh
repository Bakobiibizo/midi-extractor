#!/bin/bash

rm -rf examples/*.mid examples/*.png

rm -rf outputs/*.mid

rm -rf uploads/*.wav

rm -rf midi_generator.log


ignore_files=("checkpoints/best_model.pt" "checkpoints/last_model.pt")

checkpoint_dir="checkpoints/"

for file in $checkpoint_dir*; do
    if [[ ! "${ignore_files[@]}" =~ "${file}" ]]; then
        rm -rf "$file"
    fi
done
