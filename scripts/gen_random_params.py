#!/usr/bin/env python3
import os, random
random.seed(0)

out_dir = "/workspace/models/random"
os.makedirs(out_dir, exist_ok=True)

shapes = {
    'conv1.weight.txt': 6*1*5*5,
    'conv1.bias.txt': 6,
    'conv2.weight.txt': 16*6*5*5,
    'conv2.bias.txt': 16,
    'fc1.weight.txt': 120*256,
    'fc1.bias.txt': 120,
    'fc2.weight.txt': 84*120,
    'fc2.bias.txt': 84,
    'fc3.weight.txt': 10*84,
    'fc3.bias.txt': 10,
}

for name, count in shapes.items():
    path = os.path.join(out_dir, name)
    with open(path, 'w') as f:
        for i in range(count):
            # Xavier-like small randoms
            f.write(f"{random.uniform(-0.1, 0.1):.6f}\n")
print(f"Wrote random params to {out_dir}")
