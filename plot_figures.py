# ./plot_figures.py

import matplotlib.pyplot as plt
import numpy as np
import os


def plot_fig3(results_dir):
    # 从结果文件中加载数据
    homo_lora_data = np.load(os.path.join(
        results_dir, "homo_lora_results.npy"))
    hetero_lora_data = np.load(os.path.join(
        results_dir, "hetero_lora_results.npy"))

    # 绘制图形
    plt.figure(figsize=(8, 6))
    for i, r in enumerate([1, 5, 20, 50]):
        plt.plot(homo_lora_data[:, i], label=f"HomoLoRA r={r}")
    plt.plot(hetero_lora_data, label="HetLoRA")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Perplexity")
    plt.title("Performance of HomoLoRA and HetLoRA")
    plt.legend()
    plt.savefig("fig3.png")
    plt.close()


def plot_fig5(results_dir):
    # 从结果文件中加载数据
    full_ft_data = np.load(os.path.join(results_dir, "full_ft_results.npy"))
    homo_lora_data = np.load(os.path.join(
        results_dir, "homo_lora_results.npy"))
    hetero_lora_data = np.load(os.path.join(
        results_dir, "hetero_lora_results.npy"))

    # 绘制图形
    plt.figure(figsize=(8, 6))
    plt.plot(full_ft_data, label="Full Fine-tuning")
    plt.plot(homo_lora_data[:, 1], label="HomoLoRA r=5")
    plt.plot(homo_lora_data[:, 3], label="HomoLoRA r=50")
    plt.plot(hetero_lora_data, label="HetLoRA")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Metric")
    plt.title("Comparison of Different Methods")
    plt.legend()
    plt.savefig("fig5.png")
    plt.close()


if __name__ == "__main__":
    results_dir = "./results"
    plot_fig3(results_dir)
    plot_fig5(results_dir)
