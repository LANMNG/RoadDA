import torch
import numpy as np
from matplotlib import pyplot as plt
from torch import nn
import torch.nn.functional as F
from advent.model.deeplabv2 import get_deeplab_v2
from advent.dataset.gta5 import GTA5DataSet
from advent.dataset.cityscapes import CityscapesDataSet
from advent.domain_adaptation.config import cfg, cfg_from_file
from torch.utils import data
from tqdm import tqdm
from advent.utils.func import prob_2_entropy
from MulticoreTSNE import MulticoreTSNE as TSNE
from time import time


device = 0
NUM_CLASSES = 2
input_size = 512
MULTI_LEVEL = True
restore_from = 'ADVENT/experiments/snapshots_dpg_cnds_adv-2_nomean/GTA2Cityscapes_DeepLabv2_AdvEnt/model_56000.pth'

def load_checkpoint_for_evaluation(model, checkpoint, device):
    saved_state_dict = torch.load(checkpoint)
    model.load_state_dict(saved_state_dict)
    model.eval()
    model.cuda(device)

def data_preparation():
    source_dataset = GTA5DataSet(root=cfg.DATA_DIRECTORY_SOURCE,
                                 list_path=cfg.DATA_LIST_SOURCE,
                                 set=cfg.TRAIN.SET_SOURCE,
                                 max_iters=cfg.TRAIN.MAX_ITERS * cfg.TRAIN.BATCH_SIZE_SOURCE,
                                 crop_size=cfg.TRAIN.INPUT_SIZE_SOURCE,
                                 mean=cfg.TRAIN.IMG_MEAN)
    source_loader = data.DataLoader(source_dataset,
                                    batch_size=1,
                                    num_workers=cfg.NUM_WORKERS,
                                    shuffle=True,
                                    pin_memory=True)

    target_dataset = CityscapesDataSet(root=cfg.DATA_DIRECTORY_TARGET,
                                       list_path=cfg.DATA_LIST_TARGET,
                                       set=cfg.TRAIN.SET_TARGET,
                                       info_path=cfg.TRAIN.INFO_TARGET,
                                       max_iters=cfg.TRAIN.MAX_ITERS * cfg.TRAIN.BATCH_SIZE_TARGET,
                                       crop_size=cfg.TRAIN.INPUT_SIZE_TARGET,
                                       mean=cfg.TRAIN.IMG_MEAN)
    target_loader = data.DataLoader(target_dataset,
                                    batch_size=1,
                                    num_workers=cfg.NUM_WORKERS,
                                    shuffle=True,
                                    pin_memory=True)

    model_gen = get_deeplab_v2(num_classes=NUM_CLASSES, multi_level=MULTI_LEVEL)
    load_checkpoint_for_evaluation(model_gen, restore_from, device)
    interp = nn.Upsample(size=(input_size, input_size), mode='bilinear', align_corners=True)
    # source
    source_iter = iter(source_loader)
    feature_source = []
    for index in tqdm(range(300)):
        image, label, _, name = next(source_iter)
        pred_main = model_gen(image.cuda(device))[1]
        output = interp(pred_main)
        feature_space = prob_2_entropy(F.softmax(output)).cpu().data[0].numpy()
        feature_vector = feature_space.transpose(1, 2, 0).flatten()
        feature_vector_list = feature_vector.tolist()
        feature_source.append(feature_vector_list)
    # target
    target_iter = iter(target_loader)
    feature_target = []
    for index in tqdm(range(300)):
        image, label, _, name = next(target_iter)
        pred_main = model_gen(image.cuda(device))[1]
        output = interp(pred_main)
        feature_space = prob_2_entropy(F.softmax(output)).cpu().data[0].numpy()
        feature_vector = feature_space.transpose(1, 2, 0).flatten()
        feature_vector_list = feature_vector.tolist()
        feature_target.append(feature_vector_list)

    feature_all = []
    feature_all = feature_source + feature_target
    print(len(feature_all))
    feature_all_narry = np.array(feature_all)
    np.save("feature_no_adaptation.npy", feature_all_narry)

def main():
    print('begin tsne')
    data = np.load('feature_no_adaptation.npy')
    t0 = time()
    embeddings = TSNE(n_jobs=4).fit_transform(data)
    t1 = time()
    print("t-SNE: %.2g sec" % (t1 - t0))
    np.save('embeddings_no_adaptation', embeddings)
    vis_x = embeddings[:, 0]
    vis_y = embeddings[:, 1]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(600):
        if i<300:
            s1 = ax.scatter(vis_x[i], vis_y[i], color='#2E75B6', marker='.')
        else:
            s2 = ax.scatter(vis_x[i], vis_y[i], color='#ED7D31', marker='+')
    plt.savefig('tsne_no_daptation.png', dpi=500)
    print('figure saved')


if __name__ == '__main__':
    data_preparation()
    #main()
