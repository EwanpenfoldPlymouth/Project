# Project
Reconstruction of Image and Text using Generative Models 
This is my final year computer science project on DDPMs.
I trained my DDPM model using (CelebA) Dataset found here "https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html"

How to set up files
Downalod the files "Final_FID.py", "Final_gui.py", "Final_modules.py", "Final_utils.py" and "Final_ddpm.py" reanme the files to remove "Final_".

Set up your folder like this:
root
|--requirements.txt
|--dataset
|       |------img_align_celeba
|               |-----img_align_celeba
|                       |-----(The images of the datset saved here in jpg)
|
|--fid_data
|      |------real
|               |----(FID dataset)
|      |-------val  
|                |---dummy
|                       |---(validation dataset)
|--models
|     |----DDPM_Unconditional 
|              |-----(ckpt.pt)
|--results
|    |-----DDPM_Unconditional 
|             |--fid_fake
|                    |--(spearted results from the batch by themselves)
|             |--(batch results)
|--runs
|    |----DDPM_Unconditional 
|               |----(event files)
|
|---FID.py
|---gui.py
|---modules.py
|---utils.py
|---ddpm.py

You will need to go into the code and change the paths, to your paths. 
run requirements.txt using "pip install -r requirements.txt"
