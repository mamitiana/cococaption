import os
import torch
from torch.utils.data.dataloader import DataLoader

from clef.annotationUtils import create_target
from clef.datasetSync import ImageClefDataset
import config
from torchvision.datasets import CocoCaptions 
import torchvision.transforms as transforms
import string
from pycocotools.coco import COCO
import json
from coco.coco import get_coco_data_raw
beam_size=3

img_transform =transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

class CocoCaptionsMine(CocoCaptions):
 
    def __init__(self, root, annFile):
        CocoCaptions.__init__(self,root, annFile, transform= img_transform)
        self.pred = []

#    def __getitem__(self, index):
#        return self.__getitem__(self,index)

    def setCaption(self,index,caption):
        img_id = self.ids[index]
        temp={
            "image_id" : img_id,
            "caption": ". ".join(caption)
            }

        self.pred.append(temp)

    def exportjson(self,path):
        
        with open(path, 'w') as fp:
            json.dump(self.pred, fp)
    def __len__(self):
        return len(self.ids)

__COCO_IMG_PATH = "/home/nnyhoavy/coco"
__COCO_ANN_PATH = "/home/nnyhoavy/coco/annotations"
VAL_PATH = {'root': os.path.join(__COCO_IMG_PATH, 'val2014'),
              'annFile': os.path.join(__COCO_ANN_PATH, 'captions_val2014.json')
              }
if __name__=="__main__":
    use_gpu = True
    print("use_gpu:"+str(use_gpu))
    pathModel ="/home/nnyhoavy/cococaption/results/2017-12-20_14-12/checkpoint_epoch_3.pth.tar"

    #valDset=ImageClefDataset(config.Configuration.valImages,config.Configuration.valext,'val')
    #testDest = ImageClefDataset(config.Configuration.testImages,config.Configuration.testext,'test')
    testDest  = CocoCaptionsMine(VAL_PATH['root'],VAL_PATH['annFile'])
    valDset = testDest
    batch_size =50
    model= torch.load(pathModel)['model']
    print('len val: '+str(len(valDset)))
    print('type: '+str(type(valDset)))
    for i  in range ( len(valDset) ):
        #print(valDset[i])
        img ,_ = valDset[i]
        if i % 1000 ==0:print(i) 
        captions = model.generate(img, beam_size=beam_size)
        valDset.setCaption(i,captions)
    print(valDset.pred)
    valDset.exportjson('testOut00.json')
    