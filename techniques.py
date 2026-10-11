#----to calculate how much number of params---------
model = None
total_params = sum(p.numel() for p in model.parameters())
total_gpt2_params = total_params - sum(p.numel() for p in model.out_head.parameters())
print(f"{total_gpt2_params:,}")

#----to calculate how much memory they take or total size of model---------
total_size_bytes = total_params * 4
total_size_mb = total_size_bytes / (1024 * 1024)
print(f"total mb is {total_size_bytes:.4f}")