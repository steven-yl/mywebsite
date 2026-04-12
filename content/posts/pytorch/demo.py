import pickle
import os

# 读取 txt，从每行路径中截取 pkl 文件名，汇总存成 pkl 文件
txt_path = "/Users/gaoyuanlong/mycode/mywebsite/content/posts/pytorch/stats_cutin_frame_cases.txt"
output_pkl = "extracted_filenames.pkl"

filenames = []
with open(txt_path, "r") as f:
    for line in f:
        line = line.strip().strip('"').rstrip(",").strip('"')
        if not line or not line.endswith(".pkl"):
            continue
        filenames.append(os.path.basename(line))

with open(output_pkl, "wb") as f:
    pickle.dump(filenames, f)

print(f"Extracted {len(filenames)} filenames, saved to {output_pkl}")
for name in filenames[:5]:
    print(f"  {name}")
if len(filenames) > 5:
    print(f"  ... and {len(filenames) - 5} more")
