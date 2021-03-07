from minicity import MiniCity
from utils import get_dataloader, get_model, get_device
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

    print(model)

    # Specify the device
    device = get_device()
    print(f"____Running the model on {device}___\n")

    print("____Training the model from scratch____") if args.train else print("___Fine tuning the model____")

    for epoch in range(args.epochs):

        train_loss = train_epoch(model, data_loader["train"], optimizer, criterion, scheduler, device, args)

        eval_loss, mean_iou = eval_epoch(model, data_loader["val"], criterion, MiniCity.classLabels, MiniCity.validClasses,
                             MiniCity.mask_colors, epoch, args.save_path, args)

        print(f"({epoch}/{args.epochs}) ---> \ttrain_loss:{train_loss:.2f}, \tvalidation: {eval_loss:.2f}")

if __name__ == '__main__':
    main()
