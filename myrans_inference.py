import os
import numpy as np
import pandas as pd
import time as time 
import argparse
import pickle

from myrans_trainer import loadData


def main(args):
    modelPath = args["model_path"]
    dataDir = args["gist_dir"]
    output_dir_sub = args["output_dir"]
    

    if "binary" in modelPath:
        stage = "binary"
        dataStage = "binary"
        Labels = ["Defect"]
    else:
        Labels = ["RB","OB","PF","DE","FS","IS","RO","IN","AF","BE","FO","GR","PH","PB","OS","OP","OK"]

        if "e2e" in modelPath:
            stage = "e2e"
        else:
            stage = "defect"
        
        dataStage = "e2e"


    outputDir = os.path.join(output_dir_sub, stage)
    if not os.path.isdir(outputDir):
        os.makedirs(outputDir)
    
    gistVal, _, filesVal, _ = loadData(dataDir, "Val", dataStage)
    gistTest, _, filesTest, _ = loadData(dataDir, "Test", dataStage)
    
    with open(os.path.join(modelPath), 'rb') as fid:
        forest = pickle.load(fid) 

    start_time = time.time()

    valProb = forest.predict_proba(gistVal)
    testProb = forest.predict_proba(gistTest)

    if isinstance(valProb, list):
        valProb = np.array(valProb)
        testProb = np.array(testProb)
    if valProb.ndim == 2:
        valProb = valProb.reshape(-1, valProb.shape[0], valProb.shape[1])
        testProb = testProb.reshape(-1, testProb.shape[0], testProb.shape[1])


    expname = os.path.basename(os.path.normpath(modelPath))
    
    val_sigmoid_dict = {}
    val_sigmoid_dict["Filename"] = filesVal
    for idx, header in enumerate(Labels):
        val_sigmoid_dict[header] = valProb[idx, :, 0]

    df_val = pd.DataFrame(val_sigmoid_dict)
    df_val.to_csv(os.path.join(outputDir, "Myrans", "Myrans_{}_val_sigmoid.csv".format(expname)), sep=",", index=False)

    
    test_sigmoid_dict = {}
    test_sigmoid_dict["Filename"] = filesTest
    for idx, header in enumerate(Labels):
        test_sigmoid_dict[header] = testProb[idx,:, 0]

    df_test = pd.DataFrame(test_sigmoid_dict)
    df_test.to_csv(os.path.join(outputDir, "Myrans", "Myrans_{}_test_sigmoid.csv".format(expname)), sep=",", index=False)

    end_time = time.time()

    print("\nTime spent: {}".format(end_time-start_time))

    

if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--model_path", type=str, default=None)
    ap.add_argument("--gist_dir", type=str, default="./GISTFeatures")
    ap.add_argument("--output_dir", type=str, default="./results")

    args = vars(ap.parse_args())
    main(args)