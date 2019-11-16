from util import *
from params import *
from metric import *
from imports import *
from training.radam import *
from training.losses import *
from training.freezing import *
from training.learning_rate import *


def fit_seg(model, train_dataset, val_dataset, epochs=50, batch_size=32, use_aux_clf=False,
            warmup_prop=0.1, lr=1e-3, schedule='cosine', min_lr=1e-5, verbose=1, verbose_eval=10):

    avg_val_loss = 1000
    lr_init = lr
    model.cuda()

    params = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    encoder_params = [(n, p) for n, p in params if any(nd in n for nd in ['encoder', 'logit', 'center'])]
    opt_params = [
        {'params': [p for n, p in encoder_params if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-3},
        {'params': [p for n, p in params if not any(nd in n for nd in no_decay) and 'decoder' in n],
         'weight_decay': 0.1},
        {'params': [p for n, p in params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]

    optimizer = RAdam(opt_params, lr=lr)
    # optimizer = RAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    if schedule == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs - ceil(epochs * warmup_prop), eta_min=min_lr)
    elif schedule == 'reduce_lr':
        scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=ceil(5 / verbose_eval) - 1)

    loss_seg = lov_loss  # hck_focal_loss
    loss_clf = BCEWithLogitsLoss(reduction='mean')
    loss_clf_w = 1.

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               drop_last=True, num_workers=NUM_WORKERS)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=VAL_BS, shuffle=False, num_workers=NUM_WORKERS)

    for epoch in range(epochs):
        model.train()
        if batch_size <= 4:
            model.apply(freeze_bn)

        avg_loss = 0
        start_time = time.time()

        lr = schedule_lr(optimizer, epoch, scheduler, scheduler_name=schedule, avg_val_loss=avg_val_loss,
                         epochs=epochs, warmup_prop=warmup_prop, lr_init=lr_init, min_lr=min_lr,
                         verbose_eval=verbose_eval)

        for x, y_batch, fault_batch in train_loader:
            optimizer.zero_grad()
            y_pred, fault_pred = model(x.cuda())
            if use_aux_clf:
                loss = loss_seg(y_pred, y_batch.cuda()) + loss_clf(fault_pred,
                                                                   fault_batch.cuda().float()) * loss_clf_w
            else:
                loss = loss_seg(y_pred, y_batch.cuda())
            loss.backward()
            avg_loss += loss.item() / len(train_loader)
            optimizer.step()

        del y_pred, fault_pred
        torch.cuda.empty_cache()

        model.eval()
        avg_val_loss = 0.
        val_dice = 0.

        if (epoch + 1) % verbose_eval == 0 or (epoch + 1) == epochs:
            with torch.no_grad():
                for x, y_batch, masks_batch, fault_batch in val_loader:
                    bs, c, h, w, = masks_batch.size()

                    y_pred, fault_pred = model(x.cuda())

                    if use_aux_clf:
                        loss = loss_seg(y_pred.detach(), y_batch.cuda()) + loss_clf(fault_pred.detach(),
                                                                                    fault_batch.cuda().float()) * loss_clf_w
                    else:
                        loss = loss_seg(y_pred.detach(), y_batch.cuda())
                    avg_val_loss += loss.item() / len(val_loader)

                    probs = nn.Softmax(-1)(y_pred.permute(0, 2, 3, 1).reshape(-1, 5).detach())[:, 1:].reshape(bs, -1, c)
                    val_dice += dice_th(probs.permute(0, 2, 1).contiguous().cpu(), masks_batch) / len(val_loader)

                del probs, masks_batch, y_pred, fault_pred
                torch.cuda.empty_cache()

        elapsed_time = time.time() - start_time

        if (epoch + 1) % verbose == 0:
            elapsed_time = elapsed_time * verbose
            print(f'Epoch {epoch + 1}/{epochs}     lr={lr:.1e}     t={elapsed_time:.0f}s     loss={avg_loss:.4f}     ',
                  end='')
            if (epoch + 1) % verbose_eval == 0 or (epoch + 1) == epochs:
                print(f'dice={val_dice:.5f}     val_loss={avg_val_loss:.4f}     ', end='\n')
            #                 save_model_weights(model, f"{epoch + 1}_{model_name}", verbose=0)
            else:
                print(' ', end='\n')

        if lr < min_lr:
            print(f'Reached low enough learning rate ({min_lr:.1e}), interrupting...')
            break

            
def fit_clf(model, train_dataset, val_dataset, epochs=50, batch_size=32, use_aux_clf=False,
        warmup_prop=0.1, lr=1e-3, schedule='cosine', min_lr=1e-6, verbose=1, 
        verbose_eval=2, cp=False, model_name='model'):
    
    avg_val_loss = 1000
    best_score = 0
    lr_init = lr
    model.cuda()
    
    optimizer = RAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    if schedule == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs-ceil(epochs*warmup_prop), eta_min=min_lr)
    elif schedule == 'reduce_lr':
        scheduler = ReduceLROnPlateau(optimizer, factor=0.2, patience=ceil(5/verbose_eval)-1)
    
    loss_clf = BCEWithLogitsLoss(reduction='mean')
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=VAL_BS, shuffle=False, num_workers=NUM_WORKERS)

    for epoch in range(epochs):
        model.train()
        avg_loss = 0
        start_time = time.time()  
        
        lr = schedule_lr(optimizer, epoch, scheduler, scheduler_name=schedule, avg_val_loss=avg_val_loss, 
                         epochs=epochs, warmup_prop=warmup_prop, lr_init=lr_init, min_lr=min_lr, verbose_eval=verbose_eval)
        
        for x, y in train_loader:
            optimizer.zero_grad()
            pred = model(x.cuda())
            loss = loss_clf(pred, y.cuda().float())
            loss.backward()
            avg_loss += loss.item() / len(train_loader)
            optimizer.step()

        
        model.eval()
        avg_val_loss = 0.
        val_acc = 0
        
        if (epoch + 1) % verbose_eval == 0 or (epoch + 1) == epochs:
            for x, y in val_loader:
                pred = model(x.cuda())
                loss = loss_clf(pred.detach(), y.cuda().float())
                avg_val_loss += loss.item() / len(val_loader)

                y_pred = (nn.Sigmoid()(pred.detach()).cpu().numpy() > 0.5).astype(int)
                val_acc += accuracy_score(y, y_pred) / len(val_loader)
                
                
        elapsed_time = time.time() - start_time
        
        if cp:
            if score > best_score:
                save_model_weights(model, f"{model_name}.pt", verbose=0)
                best_score = val_dice
        
        if (epoch + 1) % verbose == 0:
            elapsed_time = elapsed_time * verbose
            print(f'Epoch {epoch+1}/{epochs}     lr={lr:.1e}     t={elapsed_time:.0f}s     loss={avg_loss:.4f}     ', end='')
            if (epoch + 1) % verbose_eval == 0 or (epoch + 1) == epochs: 
                print(f'val_loss={avg_val_loss:.4f}     val_acc={val_acc:.4f}')
            else:
                print(' ', end='\n')
        
        if (epoch+1) % 5 == 0:
            save_model_weights(model, f"{model_name}_{i+1}_{epoch+1}.pt", verbose=0)
                
        if lr < min_lr:
            print(f'Reached low enough learning rate ({min_lr:.1e}), interrupting...')
            break