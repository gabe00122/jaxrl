# uv run pmarl train --config ./config/num_layers_6.json
# uv run pmarl train --config ./config/rms_norm.json
# uv run pmarl train --config ./config/digging.json
# uv run pmarl train --config ./config/qk_norm.json
# uv run pmarl train --config ./config/post_attn_norm.json
# uv run pmarl train --config ./config/post_ffw_norm.json
# uv run pmarl train --config ./config/digging_baseline.json
# uv run pmarl train --config ./config/config-no-obs.json
# uv run pmarl train --config ./config/config.json
# uv run ./jaxrl/sweep.py

# uv run pmarl train --config ./config/blog_baseline.json

# uv run pmarl train --config ./config/blog_2_layers.json
# uv run pmarl train --config ./config/blog_4_layers.json
# uv run pmarl train --config ./config/blog_8_layers.json

# uv run pmarl train --config ./config/blog_64_width.json
# uv run pmarl train --config ./config/blog_256_width.json

# uv run pmarl train --config ./config/blog_ff_512.json
# uv run pmarl train --config ./config/blog_ff_1024.json

# uv run pmarl train --config ./config/blog_2_head.json
# uv run pmarl train --config ./config/blog_4_head.json

# uv run pmarl train --config ./config/blog_mlp.json
# uv run pmarl train --config ./config/blog_rnn.json

# uv run pmarl train --config ./config/blog_10_layers.json
# uv run pmarl train --config ./config/blog_12_layers.json
# uv run pmarl train --config ./config/blog_14_layers.json
# uv run pmarl train --config ./config/blog_16_layers.json
# uv run pmarl train --config ./config/blog_32_layers.json

uv run pmarl train --config ./config/koth.json
uv run pmarl eval --run $(ls ./results/ -Art | tail -n 1) --env koth --rounds 5000 --out ./analysis/$(ls ./results/ -Art | tail -n 1)
