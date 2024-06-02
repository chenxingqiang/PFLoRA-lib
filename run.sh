cd dataset
python generate_glue_cola.py noniid balance ""  CoLA

# 运行 HomoLoRA 实验
python main.py --dataset GLUE/CoLA   --model  HomoLoRA  --num_clients 2 --num_classes 2 --num_epochs 1 --batch_size 32  --local_epochs 1  --hetlora_gamma 0.99 --lora_alpha 16 --algorithm HomoLoRA


# 运行 HetLoRA 实验
cd  system
python main.py --dataset GLUE/CoLA   --model  HetLoRA  --num_clients 2 --num_classes 2 --num_epochs 1 --batch_size 32 --local_epochs 1 --hetlora_min_rank 2 --hetlora_max_rank 8 --hetlora_gamma 0.99 --lora_alpha 16 --lora_dropout 0.1 --algorithm HetLoRA


# 绘制图形
python plot_figures.py