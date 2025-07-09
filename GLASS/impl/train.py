import torch
from tqdm.auto import tqdm


def train(optimizer, model, dataloader, loss_fn):
    '''
    Train models in an epoch.
    '''
    model.train()
    total_loss = []

    num_batches = 0
    for batch in tqdm(dataloader):
        optimizer.zero_grad()
        pred = model(*batch[:-1], id=0)
        loss = loss_fn(pred, batch[-1])
        loss.backward()
        total_loss.append(loss.detach().item())
        optimizer.step()
        num_batches += 1

    if num_batches == 0:
        print("[Warning] No batch in dataloader!")
        return 0.0

    return sum(total_loss) / len(total_loss)



@torch.no_grad()
def test(model, dataloader, metrics, loss_fn):
    '''
    Test models either on validation dataset or test dataset.
    '''
    model.eval()
    preds = []
    ys = []
    for batch in dataloader:
        pred = model(*batch[:-1])
        preds.append(pred)
        ys.append(batch[-1])
    pred = torch.cat(preds, dim=0)
    y = torch.cat(ys, dim=0)
    return metrics(pred.cpu().numpy(), y.cpu().numpy()), loss_fn(pred, y)
