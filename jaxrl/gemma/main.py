from safetensors import safe_open

tensors = {}
with safe_open("../foundation/model.safetensors", framework="flax", device="cpu") as f:
    for i in range(26):
        key = f"model.layers.{i}.self_attn.v_proj.weight"
        print(f.get_tensor(key))
