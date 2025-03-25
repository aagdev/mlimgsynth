#/bin/sh

### Generation options
PROMPT="a photograph of an astronaut riding a horse"
NPROMPT=
SEED=
# Sizes: 512,512 512,768 768,512 1024,1024 1216,832 832,1216
SIZE=

OUTNAME="output"
OUTEXT="png"
NBATCH=1

CFG_SCALE=7
STEPS=20

# Sampling method: euler, euler_a, taylor3, dpm++2m, dpm++2s_a
METHOD=euler_a
# Scheduler: uniform, karras
SCHED=uniform
SAMPOPT=

# Leave empty to use the best (GPU usually).
#BACKEND=Vulkan0
#BACKEND=CPU

# Change to the path of the model weights
# Supported models: SD 1, 2 or XL
# Supported formats: safetensors
MODEL="../models/sd_v1.5-pruned-emaonly-fp16.safetensors"
#MODEL="../models/DreamShaper_8.safetensors"
#MODEL="../models/dreamshaperXL_v21TurboDPMSDE.safetensors"

# LoRA's
LORADIR="../models/loras_sd15"
#PROMPT="$PROMPT<lora:add_detail:0.75>"

EXTRA=
# You may enable any of the following options removing the # in front

# Reduce memory usage
#EXTRA="$EXTRA --unet-split 1 --vae-tile 512"

# Use TAE instead of VAE to decode faster and with less memory
#EXTRA="$EXTRA --tae '../models/tae_sd.safetensors'"

# Perform img2img
# Inpaints if the image has an alpha channel
#EXTRA="$EXTRA -i 'input_image.png' --f-t-ini 0.7"

# Debug output
#EXTRA="$EXTRA --debug"

# Batch generation
idx=1
while [ $idx -le $NBATCH ]; do
	echo "Generating $idx / $NBATCH"
	./mlimgsynth generate -b "$BACKEND" -m "$MODEL" --lora-dir "$LORADIR" -p "$PROMPT" -n "$NPROMPT" -o "$OUTNAME-$idx.$OUTEXT" -d "$SIZE" --cfg-scale "$CFG_SCALE" --steps "$STEPS" --seed "$SEED" --method "$METHOD" --scheduler "$SCHED" $SAMPOPT $EXTRA
	[ "$SEED" = "" ] || SEED=$(($SEED+1))
    idx=$(($idx+1))
done
