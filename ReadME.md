# Hello There :) 

This repo walks you through deploying a custom model to a sagemaker endpoint.

Start by running the **Llama_3_1_8b_+_Unsloth_2x_faster_finetuning** notebook to finetune and save the finetuned model locally. I successfully ran this using the Pytorch GPU image 2.4 and ml.g5.2xlarge instance type.

After saving the model adapter locally, run the **deploy model** notebook to package and deploy the model to an endpoint
