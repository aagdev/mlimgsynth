@echo off
title mlimgsynth

REM Generation options
set PROMPT=a photograph of an astronaut riding a horse
set NPROMPT=
set SEED=
set WIDTH=
set HEIGHT=
set OUTNAME=output
set OUTEXT=png
set NBATCH=1

set CFG_SCALE=7
set STEPS=12

REM Sampling method
set METHOD=dpm++2m
set SCHED=karras
set SAMPOPT=
REM set SAMPOPT=--s-ancestral 1

REM Leave empty to use CPU if you do not have a supported GPU
set BACKEND=Vulkan0
REM set BACKEND=

REM Change to the path of the model weights
REM Supported models: SD 1, 2 or XL
REM Supported formats: safetensors
set MODEL=../models/sd_v1.5-pruned-emaonly-fp16.safetensors
REM set MODEL=../models/DreamShaper_8.safetensors
REM set MODEL=../models/dreamshaperXL_v21TurboDPMSDE.safetensors

set EXTRA=

REM You may enable any of the following options removing the REM in front

REM Reduce memory usage
REM set EXTRA=%EXTRA% --unet-split --vae-tile 512

REM Use TAE instead of VAE to decode faster and with less memory
REM set EXTRA=%EXTRA% --tae "../models/tae_sd.safetensors"

REM Perform img2img
REM set EXTRA=%EXTRA% -i "input_image.png" --f-t-ini 0.7

REM Debug output
REM set EXTRA=%EXTRA% -d

for /L %%I in (1,1,%NBATCH%) do (
	echo Generating %%I / %NBATCH%
	mlimgsynth generate -b "%BACKEND%" -m "%MODEL%" -p "%PROMPT%" -n "%NPROMPT%" -o "%OUTNAME%-%%I.%OUTEXT%" -W "%WIDTH%" -H "%HEIGHT%" --cfg-scale "%CFG_SCALE%" --steps "%STEPS%" --seed "%SEED%" --method "%METHOD%" --sched "%SCHED%" %SAMPOPT% %EXTRA%
	if errorlevel 1 goto error
)
goto done

:error
echo ERROR %ERRORLEVEL%
:done
pause
