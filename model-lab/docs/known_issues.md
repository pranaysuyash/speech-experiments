# Known Issues and Workarounds

## Platform-Specific Issues

### Torchvision Compatibility
Installing `torchvision` can cause `torch`/`transformers` import or operator mismatches on some macOS setups (Apple Silicon). 
**Workaround**: Prefer excluding `torchvision` from model venvs unless absolutely required (e.g., for LAION-CLAP).

### MPS Multiprocessing
MPS operations may crash with `mutex lock failed` when using the default `fork` start method on macOS.
**Fix**: Set `multiprocessing.set_start_method("spawn", force=True)` before importing torch.
