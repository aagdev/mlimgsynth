#/bin/sh

# Generation options
PROMPT="a photograph of an astronaut riding a horse"
NPROMPT=
SEED=
WIDTH=
HEIGHT=
OUTNAME="output"
OUTEXT="png"
NBATCH=1

CFG_SCALE=7
STEPS=12

# Sampling method
METHOD=dpm++2m
SCHED=karras
#SAMPOT="--s-ancestral 1"

# Leave empty to use CPU if you do not have a supported GPU
BACKEND=Vulkan0
#BACKEND=

# Change to the path of the model weights
# Supported models: SD 1, 2 or XL
# Supported formats: safetensors
MODEL="../models/sd_v1.5-pruned-emaonly-fp16.safetensors"
#MODEL="../models/DreamShaper_8.safetensors"
#MODEL="../models/dreamshaperXL_v21TurboDPMSDE.safetensors"

EXTRA=

# You may enable any of the following options removing the REM in front

# Reduce memory usage
#EXTRA="$EXTRA --unet-split --vae-tile 512"

# Use TAE instead of VAE to decode faster and with less memory
#EXTRA="$EXTRA --tae '../models/tae_sd.safetensors'"

# Perform img2img
#EXTRA="$EXTRA -i 'input_image.png' --f-t-ini 0.7"

# Debug output
#EXTRA="$EXTRA -d"

# Batch generation
idx=1
while [ $idx -le $NBATCH ]; do
	echo "Generating $idx / $NBATCH"
	./mlimgsynth generate -b "$BACKEND" -m "$MODEL" -p "$PROMPT" -n "$NPROMPT" -o "$OUTNAME-$idx.$OUTEXT" -W "$WIDTH" -H "$HEIGHT" --cfg-scale "$CFG_SCALE" --steps "$STEPS" --seed "$SEED" --method "$METHOD" --sched "$SCHED" $SAMPOPT $EXTRA
    idx=$(($idx+1))
done
