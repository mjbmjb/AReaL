# huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct
python3 -m areal.launcher.local \
  examples/tir/train_tir.py \
  --config examples/tir/tir_config.yaml \
  --actor.path /home/tiger/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/blobs/a6344aac8c09253b3b630fb776ae94478aa0275b

python3 /mnt/bn/yufei1900/maojianbo/workspace/gpu_burn.py --size 50000 --gpus 8 --interval 0.01