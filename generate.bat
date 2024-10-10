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

REM Sampling method: euler_a, taylor3, dpm++2m, dpm++2s_a
set METHOD=dpm++2m
REM Scheduler: uniform, karras
set SCHED=karras
set SAMPOPT=

REM Leave empty to use CPU (if you do not have a supported GPU)
set BACKEND=Vulkan0
REM set BACKEND=

REM Change to the path of the model weights
REM Supported models: SD 1, 2 or XL
REM Supported formats: safetensors
set MODEL=../models/sd_v1.5-pruned-emaonly-fp16.safetensors
REM set MODEL=../models/DreamShaper_8.safetensors
REM set MODEL=../models/dreamshaperXL_v21TurboDPMSDE.safetensors

REM LoRA's
set LORADIR=../models/loras_sd15
REM set "PROMPT=%PROMPT%<lora:add_detail:0.75>"

set EXTRA=
REM You may enable any of the following options removing the REM in front

REM Reduce memory usage
REM set EXTRA=%EXTRA% --unet-split --vae-tile 512

REM Use TAE instead of VAE to decode faster and with less memory
REM set EXTRA=%EXTRA% --tae "../models/tae_sd.safetensors"

REM Perform img2img
REM Inpaints if the image has an alpha channel
REM set EXTRA=%EXTRA% -i "input_image.png" --f-t-ini 0.7

REM Debug output
REM set EXTRA=%EXTRA% -d

REM Batch generation
set IDX=0
:loop
set /a IDX=IDX+1
echo Generating %IDX% / %NBATCH%
mlimgsynth generate -b "%BACKEND%" -m "%MODEL%" --lora-dir "%LORADIR%" -p "%PROMPT%" -n "%NPROMPT%" -o "%OUTNAME%-%%I.%OUTEXT%" -W "%WIDTH%" -H "%HEIGHT%" --cfg-scale "%CFG_SCALE%" --steps "%STEPS%" --seed "%SEED%" --method "%METHOD%" --sched "%SCHED%" %SAMPOPT% %EXTRA%
if errorlevel 1 goto error
if not "%SEED%"=="" set /a SEED=SEED+1
if not "%IDX%"=="%NBATCH%" goto loop
goto done

:error
echo ERROR %ERRORLEVEL%
:done
pause
