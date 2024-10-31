import argparse
import logging
import os
import pprint

import torch
import numpy as np
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.semicd import SemiCDDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from model.semseg.pspnet import PSPNet
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log
from util.dist_helper import setup_distributed

parser = argparse.ArgumentParser(
    description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, default=None)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)



def evaluate(model, loader, cfg):
    model.eval()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    correct_pixel = AverageMeter()
    total_pixel = AverageMeter()

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    with torch.no_grad():
        for imgA, imgB, mask, id in loader:
            imgA = imgA.to(device)
            imgB = imgB.to(device)
            mask = mask.to(device)

            pred = model(imgA, imgB).argmax(dim=1)

            # 使用 numpy 操作交集和并集计算
            intersection, union, target = intersectionAndUnion(pred.cpu().numpy(), mask.cpu().numpy(), cfg['nclass'],
                                                               255)

            # 保证交集和并集是数组，逐类别更新
            intersection_meter.update(intersection)
            union_meter.update(union)

            correct_pixel.update((pred == mask).sum().item())
            total_pixel.update(pred.numel())

            # 计算 TP, FP, FN, TN (二分类任务)
            tp = ((pred == 1) & (mask == 1)).sum().item()
            fp = ((pred == 1) & (mask == 0)).sum().item()
            fn = ((pred == 0) & (mask == 1)).sum().item()
            tn = ((pred == 0) & (mask == 0)).sum().item()

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

    # 逐类别计算 IoU
    iou_class = np.array(intersection_meter.sum) / (np.array(union_meter.sum) + 1e-10) * 100.0

    # 打印 iou_class 类型和内容，调试用
    # print("iou_class:", iou_class, "type:", type(iou_class))

    overall_acc = correct_pixel.sum / total_pixel.sum * 100.0

    # 计算 precision, recall, F1 score
    precision = total_tp / (total_tp + total_fp + 1e-10)
    recall = total_tp / (total_tp + total_fn + 1e-10)
    f1_score = 2 * precision * recall / (precision + recall + 1e-10)

    # 计算 Kappa 系数
    oa = (total_tp + total_tn) / (total_tp + total_fp + total_tn + total_fn + 1e-10)
    pre = ((total_tp + total_fn) * (total_tp + total_fp) + (total_tn + total_fp) * (total_tn + total_fn)) / (
            (total_tp + total_fp + total_tn + total_fn + 1e-10) ** 2)
    kappa = (oa - pre) / (1 - pre + 1e-10)

    return iou_class, overall_acc, precision, recall, f1_score, kappa


def rand_bbox(size, lam=None):
    # past implementation
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception
    B = size[0]

    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(size=[B, ], low=int(W / 8), high=W)
    cy = np.random.randint(size=[B, ], low=int(H / 8), high=H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)

    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cut_mix_label_adaptive(unlabeled_image, unlabeled_mask, unlabeled_logits,
                           labeled_image, labeled_mask, lst_confidences):
    assert len(lst_confidences) == len(unlabeled_image), "Ensure the confidence is properly obtained"
    assert labeled_image.shape == unlabeled_image.shape, "Ensure shape match between labeled and unlabeled images"

    mix_unlabeled_image = unlabeled_image.clone()
    mix_unlabeled_target = unlabeled_mask.clone()
    mix_unlabeled_logits = unlabeled_logits.clone()
    labeled_logits = torch.ones_like(labeled_mask)

    u_rand_index = torch.randperm(unlabeled_image.size(0))

    # 生成相同的裁剪区域
    bbx1, bby1, bbx2, bby2 = rand_bbox(unlabeled_image.size(), lam=np.random.beta(8, 2))

    # 自适应CutMix操作
    for i in range(mix_unlabeled_image.shape[0]):
        if np.random.random() > lst_confidences[i]:
            # 变化前图像
            mix_unlabeled_image[i, :, bbx1[i]:bbx2[i], bby1[i]:bby2[i]] = \
                labeled_image[u_rand_index[i], :, bbx1[i]:bbx2[i], bby1[i]:bby2[i]]

            mix_unlabeled_target[i, bbx1[i]:bbx2[i], bby1[i]:bby2[i]] = \
                labeled_mask[u_rand_index[i], bbx1[i]:bbx2[i], bby1[i]:bby2[i]]

            mix_unlabeled_logits[i, bbx1[i]:bbx2[i], bby1[i]:bby2[i]] = \
                labeled_logits[u_rand_index[i], bbx1[i]:bbx2[i], bby1[i]:bby2[i]]

    for i in range(unlabeled_image.shape[0]):
        # 变化后图像
        unlabeled_image[i, :, bbx1[i]:bbx2[i], bby1[i]:bby2[i]] = \
            mix_unlabeled_image[u_rand_index[i], :, bbx1[i]:bbx2[i], bby1[i]:bby2[i]]

        unlabeled_mask[i, bbx1[i]:bbx2[i], bby1[i]:bby2[i]] = \
            mix_unlabeled_target[u_rand_index[i], bbx1[i]:bbx2[i], bby1[i]:bby2[i]]

        unlabeled_logits[i, bbx1[i]:bbx2[i], bby1[i]:bby2[i]] = \
            mix_unlabeled_logits[u_rand_index[i], bbx1[i]:bbx2[i], bby1[i]:bby2[i]]

    del mix_unlabeled_image, mix_unlabeled_target, mix_unlabeled_logits, labeled_logits

    return unlabeled_image, unlabeled_mask, unlabeled_logits


def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    previous_best_precision = 0.0
    previous_best_recall = 0.0
    previous_best_f1_score = 0.0
    previous_best_kappa = 0.0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))

        writer = SummaryWriter(args.save_path)

        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    model_zoo = {'deeplabv3plus': DeepLabV3Plus, 'pspnet': PSPNet}
    assert cfg['model'] in model_zoo.keys()
    model = model_zoo[cfg['model']](cfg)

    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                     {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                      'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=False)

    criterion = nn.CrossEntropyLoss(ignore_index=255).cuda(local_rank)

    trainset = SemiCDDataset(cfg['dataset'], cfg['data_root'], 'train_l', cfg['crop_size'], args.labeled_id_path)
    valset = SemiCDDataset(cfg['dataset'], cfg['data_root'], 'val')

    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=cfg['batch_size'],
                             pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler)

    iters = 0
    total_iters = len(trainloader) * cfg['epochs']
    previous_best_iou, previous_best_acc = 0.0, 0.0
    epoch = -1

    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best_iou = checkpoint['previous_best_iou']
        previous_best_acc = checkpoint['previous_best_acc']

        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)

    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info(
                '===========> Epoch: {:}, LR: {:.5f}, Previous best Changed IoU: {:.2f}, Overall Accuracy: {:.2f}'.format(
                    epoch, optimizer.param_groups[0]['lr'], previous_best_iou, previous_best_acc))

        model.train()
        total_loss = AverageMeter()

        trainsampler.set_epoch(epoch)

        for i, (imgA, imgB, mask) in enumerate(trainloader):

            imgA, imgB, mask = imgA.cuda(), imgB.cuda(), mask.cuda()

            pred = model(imgA, imgB)

            loss = criterion(pred, mask)

            torch.distributed.barrier()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())

            iters = epoch * len(trainloader) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']

            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss.item(), iters)

            if (i % (max(2, len(trainloader) // 8)) == 0) and (rank == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}'.format(i, total_loss.avg))


        iou_class, overall_acc, precision, recall, f1_score, kappa = evaluate(model, valloader, cfg)

        if rank == 0:
            logger.info('***** Evaluation ***** >>>> Unchanged IoU: {:.2f}'.format(iou_class[0]))
            logger.info('***** Evaluation ***** >>>> Changed IoU: {:.2f}'.format(iou_class[1]))
            logger.info('***** Evaluation ***** >>>> Overall Accuracy: {:.2f}'.format(overall_acc))
            logger.info('***** Evaluation ***** >>>> Precision: {:.2f}'.format(precision * 100))
            logger.info('***** Evaluation ***** >>>> Recall: {:.2f}'.format(recall * 100))
            logger.info('***** Evaluation ***** >>>> F1 Score: {:.2f}'.format(f1_score * 100))
            logger.info('***** Evaluation ***** >>>> Kappa: {:.2f}'.format(kappa * 100))

            writer.add_scalar('eval/unchanged_IoU', iou_class[0], epoch)
            writer.add_scalar('eval/changed_IoU', iou_class[1], epoch)
            writer.add_scalar('eval/overall_accuracy', overall_acc, epoch)
            writer.add_scalar('eval/precision', precision * 100, epoch)
            writer.add_scalar('eval/recall', recall * 100, epoch)
            writer.add_scalar('eval/f1_score', f1_score * 100, epoch)
            writer.add_scalar('eval/kappa', kappa * 100, epoch)

        is_best_iou = iou_class[1] > previous_best_iou
        previous_best_iou = max(iou_class[1], previous_best_iou)

        is_best_acc = overall_acc > previous_best_acc
        previous_best_acc = max(overall_acc, previous_best_acc)

        is_best_precision = precision > previous_best_precision
        previous_best_precision = max(precision, previous_best_precision)

        is_best_recall = recall > previous_best_recall
        previous_best_recall = max(recall, previous_best_recall)

        is_best_f1_score = f1_score > previous_best_f1_score
        previous_best_f1_score = max(f1_score, previous_best_f1_score)

        is_best_kappa = kappa > previous_best_kappa
        previous_best_kappa = max(kappa, previous_best_kappa)

        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best_iou': previous_best_iou,
                'previous_best_acc': previous_best_acc,
                'previous_best_precision': previous_best_precision,
                'previous_best_recall': previous_best_recall,
                'previous_best_f1_score': previous_best_f1_score,
                'previous_best_kappa': previous_best_kappa,
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best_iou:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))


if __name__ == '__main__':
    main()
