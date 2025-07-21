from pathlib import Path
from jaxrl.gemma.params import load_and_format_params
from jaxrl.gemma.transformer import Transformer

cp_path = Path("./foundation/gemma-3-flax-gemma3-1b-it-v1/gemma3-1b-it")
cp_path = cp_path.absolute().as_posix()

params = load_and_format_params(cp_path)


t = Transformer.from_params(params)
print(t)
