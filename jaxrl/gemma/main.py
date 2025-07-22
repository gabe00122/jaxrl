from pathlib import Path
import sentencepiece as spm

from jaxrl.gemma.params import load_and_format_params
from jaxrl.gemma.transformer import Transformer
from jaxrl.gemma.sampler import Sampler

cp_path = Path("./foundation/gemma-3-flax-gemma3-1b-it-v1/gemma3-1b-it")
cp_path = cp_path.absolute().as_posix()

params = load_and_format_params(cp_path)
t = Transformer.from_params(params)
del params

print(t)

sp = spm.SentencePieceProcessor(model_file="./foundation/gemma-3-flax-gemma3-1b-it-v1/tokenizer.model")

sampler = Sampler(t, sp)
output = sampler(["Write a short poem: "], 1024)

print(output.text)
