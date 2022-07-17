import logging
import torch
from torch import nn
from runx.logx import logx
from modules import CNN
from utils import init_func


def train_model(args, model, train_loader, model_type, save_path=None):
    logging.info(f'training {model_type} model in {args.dataset_name}')
    # torch.cuda.empty_cache()
    model.train()
    loss_func = nn.CrossEntropyLoss()
    correct = 0
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    if args.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    for epoch in range(1, args.epochs+1):
        for step, (x, y) in enumerate(train_loader):
            x = x.cuda()
            y = y.cuda()
            optimizer.zero_grad()
            output = model(x)
            loss = loss_func(output, y)
            loss.backward()
            optimizer.step()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).sum().item()
        if epoch > 1 and epoch % 25 == 0:
            accuracy = correct/(len(train_loader.dataset)*25)
            logging.info(f'trained {model_type} model in {args.dataset_name} epoch {epoch} with loss {loss.item()} and accuracy {accuracy}\n')
            correct = 0
    if save_path:
        # torch.save(model.state_dict(), save_path)
        torch.save(model, save_path)
    logging.info(f'trained {model_type} model in {args.dataset_name}')


# def train_target_module(args, module, train_loader, epoch, optimizer):
#     module.train()
#     loss_func = nn.CrossEntropyLoss()
#     for step, (x, y) in enumerate(train_loader):
#         x = x.cuda()
#         y = y.cuda()
#         output = module(x)
#         loss = loss_func(output, y)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         if step % args.log_interval == 0:
#             logx.msg('TargetModel Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch,
#                 step * len(train_loader),
#                 len(train_loader.dataset),
#                 100. * step / len(train_loader),
#                 loss.item()))
#
#
# def train_shadow_module(args, target_model, shadow_module, shadow_dataset_loader, epoch, optimizer):
#     target_model.train()
#     shadow_module.train()
#     dataset = args.datasets[args.dataset_ID]
#     # state_dict, _ = logx.load_model(path=args.logdir+dataset+'/'+str(3000)+'/target'+'/best_checkpoint_ep.pth')
#     # target_model = CNN('CNN7', dataset)
#     # target_model.load_state_dict(state_dict)
#     # target_model = torch.load(args.logdir+dataset+'/'+str(3000)+'/target_module.pt')
#     loss_func = nn.CrossEntropyLoss()
#     for step, (x, _) in enumerate(shadow_dataset_loader):
#         x = x.cuda()
#         _, y = target_model(x).max(1)
#         optimizer.zero_grad()
#         output = shadow_module(x)
#         loss = loss_func(output, y)
#         loss.backward()
#         optimizer.step()
#         if step % args.log_interval == 0:
#             logx.msg('ShadowModel Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch,
#                 step * len(x),
#                 len(shadow_dataset_loader.dataset),
#                 100. * step / len(shadow_dataset_loader),
#                 loss.item()))
#
#
# def save_shadow_module(args, module, epoch):
#     save_dict = {
#         'epoch': epoch + 1,
#         'state_dict': module.state_dict(),
#     }
#     logx.save_model(
#         save_dict,
#         epoch='',
#         higher_better=True
#     )