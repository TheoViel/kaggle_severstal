def get_lr(epoch):
    if epoch < 20:
        return 5e-4
    else:
        return 5e-5


def schedule_lr(
        optimizer,
        epoch,
        scheduler,
        scheduler_name='',
        avg_val_loss=0,
        epochs=100,
        warmup_prop=0.1,
        lr_init=1e-3,
        min_lr=1e-6,
        verbose_eval=1):
    if epoch <= epochs * warmup_prop and warmup_prop > 0:
        lr = min_lr + (lr_init - min_lr) * epoch / (epochs * warmup_prop)
        lrs = [lr / 10, lr / 10, lr / 4, lr / 4, lr, lr]
        for param_group, r in zip(optimizer.param_groups, lrs):
            param_group['lr'] = lr
    else:
        if scheduler_name == 'cosine':
            scheduler.step()
        elif scheduler_name == 'reduce_lr':
            if (epoch + 1) % verbose_eval == 1:
                scheduler.step(avg_val_loss)
        else:  # Manual scheduling
            lr = get_lr(epoch)
            lrs = [lr / 10, lr / 10, lr / 4, lr / 4, lr, lr]
            for param_group, r in zip(optimizer.param_groups, lrs):
                param_group['lr'] = lr
        # lr = np.mean([param_group['lr'] for param_group in optimizer.param_groups])
    lr = optimizer.param_groups[-1]['lr']
    # print(lr,epoch,epochs,epochs * warmup_prop)
    return lr