# cs234-chatbot-RLHF
cs234 final project

#Description of the project
This is the final project for CS234 called chatbot using RLHF. We introduce a method for fine-tuning large language models through Direct Preference Optimization (DPO), a reinforcement learning technique. Our experiments show that DPO streamlines the training process, enhances computational efficiency, and delivers competitive performance. Evaluation with BLEU, ROUGE, and cosine similarity metrics confirms effective learning and convergence, although additional research is required to address observed training instability.


#File explanation
RLHF_DPO.ipynb: It is the main file, focusing on fine tune the chatbot with DPO method with pairwise data
bleu_rouge.ipynb: Evaluation metrics, including BLEU and ROUGE, were used to compare the output of the fine-tuned chatbot and the pretrained chatbot against the reference answers.
cosine_sin.py: evaluation metric of cosine similarity on the output from the post-trained and pretrained chatbot compared to the reference model
