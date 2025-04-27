# Project
Reconstruction of Image and Text using Generative Models 
This is my final year computer science project on DDPMs.
I trained my DDPM model using (CelebA) Dataset found here "https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html"

How to set up files
Downalod the files "Final_FID.py", "Final_gui.py", "Final_modules.py", "Final_utils.py" and "Final_ddpm.py" rename the files to remove "Final_".

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

When the dataset has been put inside the dataset file. 
run FID.py. This will the create the FID dataset, move however many FID images from FID dataset to validation to create your valdiation dataset.  
When training dataset, FID and validation datsets have been created you can run ddpm.py file.
In the terminal you can run "python ddpm.py" to start new traing or to reume run "python ddpm.py --resume"
If you wish to change epochs, datset size, batch size these can be found at the bottom of ddpm.py. chnage them if you wish. 
