#!/usr/bin/bash

# Generate tokens using ccfx
time find . -name "*.java" | xargs -P7 -I@ /home/wuyuhao/app/ccfx/ubuntu32/ccfx d java -p @

# Remove line numbers
time find . -name "*.ccfxprep" | xargs -P7 -I@ perl -i.orig -pe 's/([^\s]+\s+){2}//' @

# Normalize identifier names
perl -pe 's/\|.*//' <filename>

# ccfx d java -p *.java
# perl -i.orig -pe 's/([^\s]+\s+){2}//' *.ccfxprep