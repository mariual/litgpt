from pathlib import Path
from typing import Literal, Optional, Union

# Everything else is reused verbatim ↓
from litgpt.finetune.lora import (
    setup as _lora_setup,
    TrainArgs, EvalArgs, DataModule,  # re-export so the CLI auto-docs stay intact
)

def setup(                     
    checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
    out_dir: Path = Path("out/finetune/qlora"),
    # ---------- QLoRA defaults ----------
    precision: Optional[str] = None,
    quantize: Literal[
        "bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq"
    ] = "bnb.nf4-dq",          # the standard QLoRA recipe
    devices: Union[int, str] = 1,
    # ------------------------------------
    **kwargs,
):
    """
    Exactly the same positional / keyword interface as `finetune.lora.setup`,
    except that:

    * `quantize` now defaults to 4-bit NF4 **with** double-quantisation
      (the canonical QLoRA configuration).
    * If the caller passes `--quantize` or `--precision` explicitly, we honour
      those just like LoRA.
    """
    return _lora_setup(        # delegate to the proven LoRA implementation
        checkpoint_dir=checkpoint_dir,
        out_dir=out_dir,
        precision=precision,
        quantize=quantize,
        devices=devices,
        **kwargs,
    )
