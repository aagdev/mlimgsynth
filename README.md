# MLImgSynth

Generate images using Stable Diffusion (SD) models. This program is completely written in C and uses the [GGML](https://github.com/ggerganov/ggml/) library. It is largely based in [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp), but with a focus in more concise and clear code. Also, I put some care in the memory usage: at each step only the required weights will be loaded in the backend memory (e.g. VRAM). Moreover, with the option `--unet-split` it is possible to run SDXL models using only 4 GiB without quantization.

## Supported models

- SD v1.x: [info](https://github.com/CompVis/stable-diffusion) [weights](https://huggingface.co/runwayml/stable-diffusion-v1-5) (`emaonly` is ok)
- SD v2.x: [info](https://github.com/Stability-AI/stablediffusion) [weights](https://huggingface.co/stabilityai/stable-diffusion-2-1)
- SDXL: [info](https://stability.ai/news/stable-diffusion-sdxl-1-announcement) [weights](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)

Besides the original weights, you may use any of the fine-tuned checkpoints that can be found on the internet.

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
./mlimgsynth generate -b CUDA0 -m MODEL.safetensors --cfg-scale 7 --steps 20 --seed 42 -o output.png -p "a box on a table"
```

The option `-b` let's you select from the available backends (by default `CPU` is used).

To start from an initial image (img2img) add the options `-i IMAGE.png` and `--f-t-ini 0.7`. The second option controls the strength by changing the initial time in the denoising process, you may try any value between 0 (no changes) and 1. 

Execute without any arguments to see a list of all the supported options.

### TAE

To accelerate and reduce the memory usage during the image decoding, you may use the [TAE](https://github.com/madebyollin/taesd) (tiny autoencoder) in place of the VAE (variational autoencoder) of SD. Download the weights compatible with SD or SDXL, and pass the path to them with the option `--tae TAE.safetensors` to enable it.

## Limitations and plans

The following are limitations of the current version that I plan to improve:

- The prompt is limited to 75 tokens and ascii characters. No syntax to change the relative weights is supported (e.g. `a (large) dog`).
- The sampling methods supported are: `euler`, `heun` and `taylor3` (change with `--method`). The last one is similar to `euler` but with two additional terms for second and third order corrections, this reduces the number of steps required. I plan to implement more methods like the popular DPM family.

## License
Most of this program is licensed under the MIT (see the file `LICENSE`), with the exceptions of the files in the directory `src/ccommon` which use the ZLib license (see the file `LICENSE.zlib`). To prevent any confusion, each file indicates its license at the beginning using the SPDX identifier.

## Contributing
Contributions in the form of bug reports, suggestions, patches or pull requests are welcome.
