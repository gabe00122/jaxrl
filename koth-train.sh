uv run pmarl train --config ./config/koth.json
uv run pmarl eval --run $(ls ./results/ -Art | tail -n 1) --env koth --rounds 5000 --out ./analysis/$(ls ./results/ -Art | tail -n 1)
