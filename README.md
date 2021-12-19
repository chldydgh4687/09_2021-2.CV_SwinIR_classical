## SwinIR: Image Restoration Using Shifted Window Transformer

Liang, J., Cao, J., Sun, G., Zhang, K., Van Gool, L. and Timofte, R., (2021. IEEE/CVF International Conference on Computer Vision)

---
##### Reference link :  https://github.com/JingyunLiang/SwinIR
##### Reference Paper : https://arxiv.org/pdf/2108.10257.pdf
##### Youtube link : 
  ---
  #### EvalAI Result
 
 - pytorch-nightly, RTX 3090 (24GB)
 - Training Time : 2 day 
 - iter : 155,000 ( 500,000 in paper )
 - Submit_File : submit.json
 - Checkpoint_path : checkpoints/Classicalx2/155000_Model.pth
   
 ![ㄴㄷㅈㄷㄱㄴㅇㄱㄴㄱㅇㄱ](https://user-images.githubusercontent.com/11037567/146667257-d748f617-387c-4878-ada6-e175bc0e48c4.png)
 
 ---
 
 #### Model Parameters
- Dataloader : RandomCrop + RandomAugmentation
- High Resolution Patch : 96
- Low Resolution Patch : 48
- batch size : 16 ( 32 in paper, because of restricted memory)  
- Learning_rate = 2e-4, gamma :  0.5

 #### Model
 - TrainDatasets : Flick2K + DIV2K
 - TestDatasets : Set5

 #### Loss Function :  L1_loss

 ---
