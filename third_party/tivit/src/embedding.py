import numpy as np
import torch
from tqdm import tqdm

from src.utils import resize_mantis_input, resize_moment_input


def embed(model, dataloader, model_type, channels, device):
    batch_embeds = []

    for (batch,) in tqdm(dataloader):

        batch_embeds_dim = []

        for dim in range(channels):
            batch_dim = batch[:, dim, :].unsqueeze(-1)

            with torch.no_grad():

                if model_type == "tivit":
                    batch_dim = batch_dim.to(device)
                    outputs = model(batch_dim).cpu().numpy()

                elif model_type == "mantis":
                    batch_dim = resize_mantis_input(batch_dim.transpose(1, 2))
                    batch_dim = batch_dim.to(device)
                    outputs = model.transform(batch_dim)

                elif model_type == "moment":
                    batch_dim = resize_moment_input(batch_dim.transpose(1, 2))
                    batch_dim = batch_dim.to(device).float()
                    outputs = model(x_enc=batch_dim).embeddings
                    outputs = outputs.cpu().numpy()

                else:
                    raise ValueError(f"Model type {model_type} unknown.")

            batch_embeds_dim.append(outputs)
            torch.cuda.empty_cache()

        batch_embeds.append(np.concatenate(batch_embeds_dim, axis=1))

    embeds = np.concatenate(batch_embeds)
    embeds /= np.linalg.norm(embeds, axis=-1, keepdims=True)

    return embeds


def concat_embeddings(
    vision_embedding_1,
    vision_embedding_2,
    mantis_embedding,
    moment_embedding,
):

    embeddings = [
        e
        for e in [
            vision_embedding_1,
            vision_embedding_2,
            mantis_embedding,
            moment_embedding,
        ]
        if e is not None
    ]

    if len(embeddings) == 0:
        raise ValueError("At least one embedding is required.")

    embeds_train = [e[0] for e in embeddings]
    embeds_test = [e[1] for e in embeddings]

    embeds_train = np.concatenate(embeds_train, axis=1)
    embeds_test = np.concatenate(embeds_test, axis=1)

    return embeds_train, embeds_test
