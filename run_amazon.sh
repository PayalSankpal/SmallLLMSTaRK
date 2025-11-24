python3 run_pipeline.py --embedding-dir ./emb --dataset amazon --split test --test-run --exp-name AMAZON

if [ -f "a.txt" ]; then
    sudo shutdown
else
    touch a.txt
fi

