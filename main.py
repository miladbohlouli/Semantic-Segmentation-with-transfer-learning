from minicity import MiniCity
from utils import get_dataloader, get_model
from arguments import get_args
from training import *
import torch
from torch import optim
from torch import nn


def main():
    args = get_args()
    model = get_model(MiniCity, args)
    data_loader = get_dataloader(MiniCity, args)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=MiniCity.voidClass)
    for param in model.parameters():
        print(param.requires_grad)

    print(model)
    print("____Training the model from scratch____") if args.train else print("___Fine tuning the model____")

    for epoch in range(args.epochs):

        train_loss = train_epoch(model, data_loader["train"], optimizer, criterion, scheduler, args)

        model.eval()
        with torch.no_grad():
            # Calculating the IoU in the specified epoch
            eval_loss, mean_iou = eval_epoch(model, data_loader["val"], criterion, MiniCity.classLabels, MiniCity.validClasses,
                                 MiniCity.mask_colors, epoch, args.save_path, args)

        print(f"({epoch}/{args.epochs}) ---> \ttrain_loss:{train_loss:.2f}, \tvalidation: {eval_loss:.2f}")

if __name__ == '__main__':
    main()
