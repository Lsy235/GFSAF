import torch
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import argparse
import os
import time

from utils.granular_loss import MultiviewGCLoss
from utils.tools import Logger
from networks.network import Network
from metric import valid
from utils.loss import LTwoLoss, SimLoss
from dataset.dataloader import load_data
from utils.train_epoches import SplitExtract, ViewsFusion

# python train.py --epochs 100 --iteration 5
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def getConfig():
    parser = argparse.ArgumentParser(description='training_args_setting')
    # hyper-params
    parser.add_argument("--lambda1", default=0.3)              # 0.3
    parser.add_argument("--lambda2", default=0.01)              # 0.3
    #data args
    parser.add_argument('--dataset', default="WebKB")
    parser.add_argument("--class_num", default=None)
    parser.add_argument("--view", default=3, type=int)
    parser.add_argument("--dims", default=None)
    parser.add_argument("--data_size", default=None, type=int)
    # model
    parser.add_argument("--device", default=None)
    parser.add_argument("--contrastive_loss", default='InfoNCE', type=str)  # 0, 1, 2
    parser.add_argument("--backbone", default='AE', type=str)      # 'AE', 'DAE'
    parser.add_argument("--iteration", default=4, type=int)         # 4
    parser.add_argument("--learning_rate", default=0.00005, type=float)         # 0.0003
    parser.add_argument("--weight_decay", default=0., type=float)              # 0.
    parser.add_argument("--ori_epochs", default=100, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--feature_dim", default=256, type=int)
    parser.add_argument("--high_feature_dim", default=64, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--p", default=2)
    parser.add_argument('--batch_size', default=256, type=int)     # 256

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    # print(device)

    return args

def main(args):
    device = args.device
    # log_name = args.exp_name  # +"_"+timeStr
    now = time.localtime()
    timeStr = str(now.tm_year) + "-" + str(now.tm_mon) + "-" + str(now.tm_mday) + "-" + str(
        now.tm_hour) + "-" + str(
        now.tm_min)
    if not os.path.isdir("./logs"):
        os.mkdir("./logs")
    logger = Logger(f'./logs/{timeStr}.txt', 'w')
    logger.log(args)

    Total_epochs = args.epochs * args.iteration


    accs = []
    purs = []
    ACC_tmp = 0

    for experIte in range(1):   # 10
        logger.log("experment iteration:{}".format(experIte+1))

        t1 = time.time()
        dataset, dims, view, data_size, class_num = load_data(args.dataset)
        args.dims = dims
        args.data_size = data_size
        args.class_num = class_num
        args.view = view

        logger.log(f"data size: {data_size}, views: {view}, class: {class_num}")
        data_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=True,
                # drop_last=True,
                drop_last=False,
            )
        # loss function
        criterion_gra = MultiviewGCLoss(args = args)
        criterion_LTwo = LTwoLoss(args.batch_size)
        criterion_Sim = SimLoss(args.batch_size)
        lossFunList = [criterion_gra, criterion_LTwo, criterion_Sim]
        tsneFea=None

        if not os.path.exists('./models'):
            os.makedirs('./models')

        model = Network(view, dims, args.feature_dim, args.high_feature_dim, class_num, device)
        # print(model)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.ori_epochs+Total_epochs, eta_min=0.)
        # max_acc = 0.0
        max_records = [[0.0, None, None, None], # multi
                       [0.0, None, None, None], # single
                       [0.0, None, None, None]] # mean

        print("Initial......")
        logger.log("Initial......")
        epoch = 0
        while epoch < args.ori_epochs:
            epoch += 1
            optimizer, lossFunList, scheduler = SplitExtract(epoch, args.backbone, args, model, optimizer, data_loader, lossFunList, scheduler, logger)

        acc, pur, saveFlag, max_records, resFea, labels_vector = valid(model, device, dataset, view, data_size, class_num, max_records=max_records)
        if(saveFlag):
            torch.save(
                {
                    "ori_epochs": epoch,
                    "Total_epochs": -1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join("models", f"{args.dataset}_best.pth"),
            )
            tsneFea = resFea
            logger.log(f"mse_epoch: {epoch}, Best Model saved.")
            print(f"mse_epoch: {epoch}, Best Model saved.")
        logger.log("=============Views Fusion Stage Start.=============")
        Iteration = 1
        logger.log(f"----------------Iter {Iteration}--------------")
        epoch = 0
        while epoch < Total_epochs:
            epoch += 1

            optimizer, scheduler = ViewsFusion(epoch, args.backbone, args, model, optimizer, data_loader, scheduler, logger)


            if epoch % args.epochs == 0:
            # if epoch % 1 == 0:
                if epoch == args.ori_epochs + Total_epochs:
                    break

                acc, pur, saveFlag, max_records, resFea, labels_vector = valid(model, device, dataset, view, data_size,
                                                                           class_num, max_records=max_records)
                if (saveFlag):
                    torch.save(
                        {
                            "ori_epochs": -1,
                            "Total_epochs": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                        },
                        os.path.join("models", f"{args.dataset}_best.pth"),
                    )
                    tsneFea = resFea
                    logger.log(f"Total_con_epoch: {epoch}, Best Model saved.")
                    print(f"Total_con_epoch: {epoch}, Best Model saved.")

                if epoch < Total_epochs:
                    Iteration += 1
                    print("Iteration " + str(Iteration) + ":")
                    logger.log("Iteration " + str(Iteration) + ":")

            pg = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(pg, lr=args.learning_rate, weight_decay=args.weight_decay)

        accs.append(acc)
        purs.append(pur)

        # if acc > ACC_tmp:
        #     ACC_tmp = acc
        #     state = model.state_dict()
        #     torch.save(state, './models/' + args.dataset + '.pth')

        t2 = time.time()
        print("Time cost: " + str(t2 - t1))
        print('End......')
        logger.log("Time cost: " + str(t2 - t1))
        logger.log('End......')

    import scipy.io as scio
    labels_vector = np.array(labels_vector)
    if not os.path.exists('./tSNE'):
        os.makedirs('./tSNE')
    # print(f"{args.dataset} resFea shape: {tsneFea.shape}, labels_vector shape: {labels_vector.shape}")
    scio.savemat(f'./tSNE/{args.dataset}.mat', {'X': tsneFea, 'Y': labels_vector})
    logger.log('Multi: ACC = {:.4f} PUR={:.4f}'.format(max_records[0][0], max_records[0][3]))
    logger.log('Single: ACC = {:.4f} PUR={:.4f}'.format(max_records[1][0], max_records[1][3]))
    logger.log('Mean: ACC = {:.4f} PUR={:.4f}'.format(max_records[2][0], max_records[2][3]))

    # print('Multi: ACC = {:.4f} NMI = {:.4f} ARI = {:.4f} PUR={:.4f}'.format(max_records[0][0], max_records[0][1], max_records[0][2], max_records[0][3]))
    # print('Single: ACC = {:.4f} NMI = {:.4f} ARI = {:.4f} PUR={:.4f}'.format(max_records[1][0], max_records[1][1], max_records[1][2], max_records[1][3]))
    # print('Mean: ACC = {:.4f} NMI = {:.4f} ARI = {:.4f} PUR={:.4f}'.format(max_records[2][0], max_records[2][1], max_records[2][2], max_records[2][3]))

if __name__ == "__main__":
    args = getConfig()
    main(args)
