# MLImgSynth

Generate images using Stable Diffusion (SD) models. This program is completely written in C and uses the [GGML](https://github.com/ggerganov/ggml/) library. It is largely based in [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp), but with a focus in more concise and clear code. Also, I put some care in the memory usage: at each step only the required weights will be loaded in the backend memory (e.g. VRAM). Moreover, with the options `--unet-split` and `--vae-tile` it is possible to run SDXL models using only 4 GiB without quantization.

## Supported models

- SD v1.x: [info](https://github.com/CompVis/stable-diffusion) [weights](https://huggingface.co/runwayml/stable-diffusion-v1-5) (`emaonly` is ok)
- SD v2.x: [info](https://github.com/Stability-AI/stablediffusion) [weights](https://huggingface.co/stabilityai/stable-diffusion-2-1)
- SDXL: [info](https://stability.ai/news/stable-diffusion-sdxl-1-announcement) [weights](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)

Besides the original weights, you may use any of the fine-tuned checkpoints that can be found on the internet. Destilled models (turbo, hyper, lightning) should work normally.

## Usage on Windows

Download and unzip the latest release. Edit the file `generate.bat` as needed and execute it.

## Build

First you must build ggml as library with the desired backends and then build this program linking to it. You may symlink the ggml directory to root of this project or define the `GGML_INCLUDE_PATH` and `GGML_LIB_PATH` variables. Finally, just call `make`. For example:

```shell
export GGML_INCLUDE_PATH=../ggml/include
export GGML_LIB_PATH=../ggml/Release/src
make
```

By default, the program is linked with `libpng` and `libjpeg` to support those formats. You may suppress these dependencies defining `MLIS_NO_PNG` and `MLIS_NO_JPEG`. The PNM image format is always available.

## Usage

First, download the weights of the model you wish to use. Right now, the only supported format is `safetensors`. To generate an image (txt2img) use:

```shell
./mlimgsynth generate -b Vulkan0 -m MODEL.safetensors --cfg-scale 7 --steps 20 --seed 42 -o output.png -p "a box on a table"
```

The option `-b` let's you select from the available backends. Use `Vulkan0` or `CUDA0` for GPU. By default `CPU` is used.

See the script `generate.sh` for a more complete example.

Execute without any arguments to see a list of all the supported options.

### img2img and inpainting

To start from an initial image (img2img) add the options `-i IMAGE.png` and `--f-t-ini 0.7`. The second option controls the strength by changing the initial time in the denoising process, you may try any value between 0 (no changes) and 1. 

If the image has an alpha channel (transparency), it is used as a mask for inpainting. You can modify the alpha channel of an image using an editor like GIMP (remember to tick the option "Save color values from transparent pixels" when saving).

### Lora's

Lora's can be loaded indivually with the option `--lora PATH:MULT` or with the option `--lora-dir PATH` and adding to the prompt `<lora:NAME:MULT>`. In the last case, it will look for the file `PATH/NAME.safetensors`.

### TAE

To accelerate and reduce the memory usage during the image decoding, you may use the [TAE](https://github.com/madebyollin/taesd) (tiny autoencoder) in place of the VAE (variational autoencoder) of SD. Download the weights compatible with SD or SDXL, and pass the path to them with the option `--tae TAE.safetensors` to enable it. Be warned that this reduces the final images quality. If you are low on memory, it is preferable to use the `--vae-tile 512` option.

## Library

All the important fuctionally is a library (libmlimgsynth) that you can use from your own programs. There are examples for C (`src/demo_mlimgsynth.c`) and for python (`python/mlimgsynth.py` and `python/guessing_game.py`).

## Future plans

- Complete prompt preprocessing to parse weights (e.g. `a (large) dog`).
- Support for GGUF and quantized models.
- API server and minimal web UI.
- ControlNet.
- Maybe SDE sampling. The biggest hurdle is understanding what it is doing the `torchsde.BrownianTree` used in `k-diffusion`.
- Other models?

## License
Most of this program is licensed under the MIT (see the file `LICENSE`), with the exceptions of the files in the directory `src/ccommon` which use the ZLib license (see the file `LICENSE.zlib`). To prevent any confusion, each file indicates its license at the beginning using the SPDX identifier.

## Contributing
Contributions in the form of bug reports, suggestions, patches or pull requests are welcome.
