python3 run_pipeline.py --embedding-dir ./emb --dataset mag --split test --test-run --exp-name MAG

if [ -f "a.txt" ]; then
    sudo shutdown
else
    touch a.txt
fi

