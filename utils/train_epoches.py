import torch

from .granular import MVGBList, GBList

def SplitExtract(epoch, rec, args, model, optimizer, data_loader, lossFunList, scheduler, logger):
    device = args.device
    view = args.view
    tot_loss = 0.
    all_loss = [0., 0., 0., 0.]
    mes = torch.nn.MSELoss()
    criterion_gra, criterion_LTwo, criterion_Sim = lossFunList[0], lossFunList[1], lossFunList[2]

    for batch_idx, (xs, y, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)

        if rec == 'AE':
            optimizer.zero_grad()
            _, lHZs, _, xLNs, xLPss, _, hs, zNoises, zPrivates = model(xs)
        if rec == 'DAE':
            noise_x = []
            for v in range(view):
                # print(xs[v])
                noise = torch.randn(xs[v].shape).to(device)
                # print(noise)
                noise = noise + xs[v]
                # print(noise)
                noise_x.append(noise)
            optimizer.zero_grad()
            _, lHZs, _, xLNs, xLPss, _, hs, zNoises, zPrivates = model(noise_x)

        loss_list = []
        loss_LTwo_list = []
        loss_far_list = []
        temp = []

        for v in range(view):
            temp.append(y)

            loss_far = criterion_Sim.forward_orthogonal(zNoises[v], zPrivates[v])
            loss_far = loss_far * loss_far
            loss_far_list.append(args.lambda1 * loss_far)

            ww = 1.0 / view
            for w in range(view):
                loss_list.append(ww * mes(xs[v], xLPss[w][v]))

            loss_LTwo_list.append(args.lambda2 * criterion_LTwo.foward_ZLTwo(zNoises[v]))
        if args.p > 1:
            # 对本批次的原数据构建粒球
            mv_gblist = MVGBList(hs, y, view)
            # 计算多视图粒球对比损失
            batch_size = y.shape[0]
            # for i in range(view):
            #     temp = mv_gblist[i].get_centers()
            #     print(f"temp shape: {temp.shape}")
            k = batch_size // view
            loss_con = criterion_gra(mv_gblist, view, k, batch_size)

            # mv_gblist = MVGBList(lHZs, y, 2)
            # # 计算多视图粒球对比损失
            # k = batch_size // 2
            # loss_LHZcon = criterion_gra(mv_gblist, view, k, batch_size)
        # if True:
        # # 对本批次的原数据构建粒球
        # mv_gblist = MVGBList(hs, y, view)
        # # 计算多视图粒球对比损失
        # batch_size = y.shape[0]
        # k = batch_size // view
        # loss_con = criterion_gra(mv_gblist, view, k, batch_size)
        #
        # p = 8
        # y_all = torch.stack(temp, dim=0)
        # yTol = y_all.reshape((-1, y_all.shape[-1]))
        # hs_all = torch.stack(hs, dim=0)
        # hsTol = hs_all.reshape((-1, hs_all.shape[-1]))
        # tol_gblist = GBList(hsTol, yTol, p=p)
        # batch_size = y.shape[0]
        # k = batch_size * view // p
        # loss_con = criterion_gra(tol_gblist, view, k, batch_size, mode=1)

        # p = 2
        # y_all = torch.stack(temp, dim=0)
        # yTol = y_all.reshape((-1,y_all.shape[-1]))
        # hs_all = torch.stack(lHZs, dim=0)
        # hsTol = hs_all.reshape((-1,hs_all.shape[-1]))
        # tol_gblist = GBList(hsTol, yTol, p=p)
        # batch_size = y.shape[0]
        # k = batch_size * view // p
        # loss_con += criterion_gra(tol_gblist, view, k, batch_size, mode = 1)
        loss_reg = sum(loss_list)
        loss_LTwo = sum(loss_LTwo_list)
        loss_far = sum(loss_far_list)
        loss = loss_reg
        loss += loss_con
        # loss += loss_Z
        # loss += loss_LHZ
        # loss += loss_LHZcon
        loss += loss_LTwo
        loss += loss_far
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
        all_loss[0] += loss_con.item()
        all_loss[1] += loss_reg.item()
        # all_loss[2] += loss_Z.item()
        # all_loss[2] += loss_LHZcon.item()
        # all_loss[2] += loss_maeCon.item()
        all_loss[2] += loss_LTwo.item()
        # all_loss[3] += loss_LHZ.item()
        all_loss[3] += loss_far.item()
    scheduler.step()
    lossFunList = [criterion_gra, criterion_LTwo, criterion_Sim]

    logger.log('Epoch {}'.format(epoch), 'Loss:{:.6f}. aL[0]:{:.6f}, aL[1]:{:.6f}, aL[2]:{:.6f}, aL[3]:{:.6f}'.format(
        tot_loss / len(data_loader),
        all_loss[0] / len(data_loader),
        all_loss[1] / len(data_loader),
        all_loss[2] / len(data_loader),
        all_loss[3] / len(data_loader)
    ))

    return optimizer, lossFunList, scheduler


def ViewsFusion(epoch, rec, args, model, optimizer, data_loader, scheduler, logger):
    device = args.device
    view = args.view
    tot_loss = 0.
    all_loss = [0., 0., 0.]
    mes = torch.nn.MSELoss()


    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)

        if (rec == 'AE'):
            optimizer.zero_grad()
            zs, lHZs, hHZs, xLNs, xLPss, xHPs, hs, zNoises, zPrivates = model(xs)
        loss_list = []

        if rec == 'DAE':
            noise_x = []
            for v in range(view):
                # print(xs[v])
                noise = torch.randn(xs[v].shape).to(device)
                # print(noise)
                noise = noise + xs[v]
                # print(noise)
                noise_x.append(noise)
            optimizer.zero_grad()
            zs, lHZs, hHZs, xLNs, xLPss, xHPs, hs, zNoises, zPrivates = model(noise_x)

        for v in range(view):
            loss_list.append(mes(xs[v], xHPs[v]))
            # ww = 1.0 / view
            # for w in range(view):
            #     loss_list.append(ww * mes(xs[v], xHPss[w][v]))

        loss_tol = sum(loss_list)
        loss = loss_tol
        # loss += loss_con
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
        all_loss[0] += loss_tol
        # all_loss[1] += loss_reg

    scheduler.step()
    # print('Epoch {}'.format(epoch), 'Loss:{:.6f}. loss_con:{:.6f}, loss_reg:{:.6f}'.format(
    #     tot_loss / len(data_loader),
    #     all_loss[0] / len(data_loader),
    #     all_loss[1] / len(data_loader)
    # ))
    logger.log('Epoch {}'.format(epoch), 'Loss:{:.6f}. loss_tol:{:.6f}'.format(
        tot_loss / len(data_loader),
        all_loss[0] / len(data_loader),
    ))


    return optimizer, scheduler
