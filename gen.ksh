# fairseq-generate data-bin/iwslt14.tokenized.de-en \
#     --path checkpoints/checkpoint_best.pt \
#     --batch-size 128 --remove-bpe --log-format json \
#     --sampling --temperature 1 --sampling-topk 0 --sampling-topp 1 \
#     --beam 1 --results-path softmax_results