import sys
sys.path.append('/home/misa/TF-mRNN-master/coco-caption-master')
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap  
valpath= "/home/misa/TF-mRNN-master/coco-caption-master/annotations/captions_val2014.json"
predpath= "/home/misa/testOut00.json"
def coco_val_eval( pred_path):
    """Evaluate the predicted sentences on MS COCO validation."""
    coco = COCO(valpath)
    #print("coco val loaded")
    #print(len(coco))
    cocoRes = coco.loadRes(pred_path)
    print("coco pred loaded")
    
    cocoEval = COCOEvalCap(coco, cocoRes)
    print("evaluating")
    
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()
    #with open("outCocoEval.json", 'w') as fout:
    for metric, score in cocoEval.eval.items():
        content = str(metric)+"   "+str(score)
        print(content)
            #print('%s: %.3f' % (metric, score), file=fout)
import json
if __name__=="__main__":
    #coco_val_eval(predpath)
    valjson = json.load(   open(valpath) )
    
    #predjson = json.load(open(predpath))

    #print(len(valjson))
    #print len(predjson) 
    #print len( valjson['annotations'] )
    print(valjson['annotations'][0])
    #print predjson[0] 