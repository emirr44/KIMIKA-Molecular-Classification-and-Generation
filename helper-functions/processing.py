import torch

def getGraphEmbeddings(model, device, dataset):
    model.eval()
    embeddings = []

    for data in dataset:
        data = data.to(device)
        with torch.no_grad():
            # Pass all required args (x, edge_index, edge_attr, batch)
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        embeddings.append(out.cpu())

    return torch.cat(embeddings, dim=0)
