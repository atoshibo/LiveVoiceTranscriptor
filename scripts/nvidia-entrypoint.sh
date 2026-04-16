#!/bin/bash
# Discover NVIDIA pip-installed library paths and export LD_LIBRARY_PATH
# so ctranslate2 / faster-whisper can dlopen libcublas.so.12 etc.
if [ -d /usr/local/lib/python3.11/site-packages/nvidia ]; then
    NVIDIA_LIBS=$(find /usr/local/lib/python3.11/site-packages/nvidia \
        -name "*.so*" -path "*/lib/*" 2>/dev/null \
        | xargs -I{} dirname {} | sort -u | tr '\n' ':' | sed 's/:$//')
    if [ -n "$NVIDIA_LIBS" ]; then
        export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
fi
exec "$@"
