# IEEE 论文源码说明

主文件：

```bash
paper/main.tex
```

参考文献：

```bash
paper/refs.bib
```

推荐编译：

```bash
cd paper
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

实验与绘图流程：

1. 先在仓库根目录训练 PPO/HE-PPO：

```bash
python run_train.py \
  --algos ppo,heppo \
  --M 10 \
  --K 2 \
  --target-angles=-40,0,40 \
  --sinr-db 12 \
  --episode-steps 8 \
  --updates 300 \
  --episodes-per-update 256 \
  --ppo-epochs 5 \
  --minibatch-size 512 \
  --lr 3e-4 \
  --action-scale 0.03 \
  --seeds 1,2,3 \
  --device cuda
```

2. 评估训练好的 RL checkpoints 并保存图数据：

```bash
python run_eval.py \
  --log-dir log/YYYYMMDD-HHMMSS \
  --eval-channels 256 \
  --plot-seed 2026 \
  --save-plots \
  --device cpu
```

3. 如需重新绘图：

```bash
python run_plot.py --log-dir log/YYYYMMDD-HHMMSS
```

4. 编译论文。
