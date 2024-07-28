# 作用:用于模型训练过程中的过拟合(fitting)或训练(training)的使用工具函数
# 1、模型训练，后期在项目中创建的train.py会调用当前的py文件，所以要比那些一些逻辑
# (1)函数通过遍历训练数据集，执行向前传播计算损失，然后执行反向传播更新模型参数，混合精度训练（fp16参数控制）
# (2)验证集评估,在每个训练周期结束后,函数会切换到验证模式,并使用验证集评估模型的性能,有助于调整训练策略
# (3)日志记录和进度显示:使用tqdm库来显示训练进度和当前损失,同时在训练过程中打印有用信息
# (4)模型保存,根据预设条件(每n个epoch保存一次或保存最佳模型)函数会保存模型的权重,有助于在训练过程中进行断点和模型部署
# (5)回调支持,通过eval_callback参数,函数支持自定义回调函数,这些函数可以在每个epoch结束时执行额外的操作
import os
import torch
from tqdm import tqdm

from utils.utils import get_lr


def fit_one_epoch(
    model_train,
    model,
    ema,
    yolo_loss,
    loss_history,
    eval_callback,
    optimizer,
    epoch,
    epoch_step,
    epoch_step_val,
    gen,
    gen_val,
    Epoch,
    cuda,
    fp16,
    scaler,
    save_period,
    save_dir,
    local_rank=0,
):
    loss = 0
    val_loss = 0

    if local_rank == 0:
        print("Start Training")
        pbar = tqdm(
            total=epoch_step,
            desc=f"Epoch {epoch + 1}/{Epoch}",
            postfix=dict,
            mininterval=0.3,
        )

    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        images, bboxes = batch  # 正确解包 batch
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                bboxes = bboxes.cuda(local_rank)

        optimizer.zero_grad()
        if not fp16:
            outputs = model_train(images)
            loss_value = yolo_loss(outputs, bboxes)
            loss_value.backward()
            torch.nn.utils.clip_grad_norm(model_train.parameters(), max_norm=10.0)
            optimizer.step()
        else:
            from torch.cuda.amp import autocast

            with autocast():
                outputs = model_train(images)
                loss_value = yolo_loss(outputs, bboxes)

            scaler.scale(loss_value).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm(model_train.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
        if ema:
            ema.update(model_train)

        loss += loss_value.item()

        if local_rank == 0:
            pbar.set_postfix(
                **{"loss": loss / (iteration + 1), "lr": get_lr(optimizer)}
            )
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print("Finish Training")
        print("Start Validation")
        pbar = tqdm(
            total=epoch_step_val,
            desc=f"Epoch {epoch + 1}/{Epoch}",
            postfix=dict,
            mininterval=0.3,
        )

    if ema:
        model_train_eval = ema.ema
    else:
        model_train_eval = model_train.eval()

    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, bboxes = batch  # 正确解包 batch
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                bboxes = bboxes.cuda(local_rank)

            optimizer.zero_grad()
            outputs = model_train_eval(images)
            loss_value = yolo_loss(outputs, bboxes)
        val_loss += loss_value.item()

        if local_rank == 0:
            pbar.set_postfix(
                **{"val_loss": val_loss / (iteration + 1), "lr": get_lr(optimizer)}
            )
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print("Finish Validation")
        loss_history.append_loss(
            epoch + 1, loss / epoch_step, val_loss / epoch_step_val
        )

        eval_callback.on_epoch_end(epoch + 1, model_train_eval)
        print("Epoch:" + str(epoch + 1) + "/" + str(Epoch))

        print(
            "Total Loss: %.3f || Val Loss: %.3f "
            % (loss / epoch_step, val_loss / epoch_step_val)
        )

        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()

        if (epoch + 1) % save_period == 0 or (epoch + 1) == Epoch:
            torch.save(
                save_state_dict,
                os.path.join(
                    save_dir,
                    "ep%3d-loss%.3f-val_loss%.3f.pth"
                    % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val),
                ),
            )

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(
            loss_history.val_loss
        ):
            print("Save best model to best_epoch_weights.pth")
            torch.save(
                save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth")
            )

        torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))


# def fit_one_epoch(model_train, model,ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
#     loss = 0
#     val_loss = 0
#
#     if local_rank == 0:
#         print('Start Training')
#         pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
#
#     model_train.train()
#     for iteration, batch in enumerate(gen):
#         if iteration >= epoch_step:
#             break
#         images, bboxes = batch[0]
#         with torch.no_grad():
#             if cuda:
#                 images = images.cuda(local_rank)
#                 bboxes = bboxes.cuda(local_rank)
#
#     optimizer.zero_grad()
#     if not fp16:
#         outputs = model_train(images)
#         loss_value = yolo_loss(outputs,bboxes)
#         loss_value.backward()
#         torch.nn.utils.clip_grad_norm(model_train.parameters(),max_norm=10.0)
#         optimizer.step()
#     else:
#         from torch.cuda.amp import autocast
#         with autocast():
#             outputs = model_train(images)
#             loss_value = yolo_loss(outputs,bboxes)
#
#         scaler.scale(loss_value).backward()
#         scaler.unscale_(optimizer)
#
#         torch.nn.utils.clip_grad_norm(model_train.parameters(),max_norm=10.0)
#         scaler.step(optimizer)
#         scaler.update()
#     if ema:
#         ema.update(model_train)
#
#     loss += loss_value.item()
#
#     if local_rank == 0:
#         pbar.set_postfix(**{'loss': loss / (iteration + 1),
#                             'lr':get_lr(optimizer)})
#         pbar.update(1)
#
#     if local_rank == 0:
#         pbar.close()
#         print('Finish Training')
#         print('Start Validation')
#         pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
#
#     if ema:
#         model_train_eval = ema.ema
#     else:
#         model_train_eval = model_train.eval()
#
#     for iteration, batch in enumerate(gen_val):
#         if iteration >= epoch_step_val:
#             break
#         images, bboxes = batch[0], batch[1]
#         with torch.no_grad():
#             if cuda:
#                 images = images.cuda(local_rank)
#                 bboxes = bboxes.cuda(local_rank)
#
#             optimizer.zero_grad()
#             outputs = model_train_eval(images)
#             loss_value = yolo_loss(outputs,bboxes)
#         val_loss +=loss_value.item()
#
#         if local_rank == 0:
#             pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1),
#                                 'lr': get_lr(optimizer)})
#             pbar.update(1)
#
#     if local_rank == 0:
#         pbar.close()
#         print('Finish Validation')
#         loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
#
#         eval_callback.on_epoch_end(epoch+1,model_train_eval)
#         print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
#
#         print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
#
#         if ema:
#             save_state_dict = ema.ema.state_dict()
#         else:
#             save_state_dict = model.state_dict()
#
#         if (epoch + 1) % save_period == 0 or (epoch + 1) == Epoch:
#             torch.save(save_state_dict, os.path.join(save_dir, "ep%3d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))
#
#         if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
#             print('Save best model to best_epoch_weights.pth')
#             torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))
#
#         torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))
