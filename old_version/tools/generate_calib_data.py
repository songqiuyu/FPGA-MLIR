import os
import numpy as np
import argparse

# 将根目录加到 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def generate_random_calib_data(output_dir, num_samples, shape):
    os.makedirs(output_dir, exist_ok=True)
    
    # 为了避免混淆，先清理已有文件
    for f in os.listdir(output_dir):
        if f.endswith(".npy"):
            os.remove(os.path.join(output_dir, f))
            
    print(f"========== [COA-MLIR] 校准数据生成 ==========")
    print(f"目标目录: {output_dir}")
    print(f"样本数量: {num_samples}")
    print(f"特征形状: {shape}")
    
    generated_files = []
    # 模拟真实图片归一化后的截断正态分布特征图 (-2.0, 2.0之间)
    for i in range(num_samples):
        # np.random.randn 是标准正态分布，适合模拟经过标准化处理的图像张量
        tensor = np.random.randn(*shape).astype(np.float32) 
        
        file_name = f"calib_batch_{i:03d}.npy"
        save_path = os.path.join(output_dir, file_name)
        np.save(save_path, tensor)
        generated_files.append(file_name)
        
    print(f"[Done] 成功生成并写入了 {len(generated_files)} 个 .npy 校准文件！\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate random calibration data")
    parser.add_argument("--samples", type=int, default=20, help="Number of calibration batches")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--channel", type=int, default=3, help="Channel size")
    parser.add_argument("--height", type=int, default=224, help="Height")
    parser.add_argument("--width", type=int, default=224, help="Width")
    
    args = parser.parse_args()
    
    calib_env_dir = os.path.join(project_root, "datasets", "calibration_data")
    shape = (args.batch, args.channel, args.height, args.width)
    
    generate_random_calib_data(calib_env_dir, args.samples, shape)
