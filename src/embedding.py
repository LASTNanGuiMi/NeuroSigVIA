import numpy as np
import torch
from tqdm import tqdm

from src.utils import resize_mantis_input, resize_moment_input


def embed(model, dataloader, model_type, channels, device):
    batch_embeds = []

    for (batch,) in tqdm(dataloader):

        if model_type == "neurosigvia" and getattr(model, "image_mode", None) in {
            "multichannel_line_plot",
            "activity_graph",
            "med_activity_graph",
            "activity_matrix",
        }:
            with torch.no_grad():
                batch = batch.to(device)
                outputs = model(batch).cpu().numpy()

            batch_embeds.append(outputs)
            torch.cuda.empty_cache()
            continue

        batch_embeds_dim = []

        for dim in range(channels):
            batch_dim = batch[:, dim, :].unsqueeze(-1)

            with torch.no_grad():

                if model_type == "neurosigvia":
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
    norms = np.linalg.norm(embeds, axis=-1, keepdims=True)
    embeds = embeds / np.maximum(norms, 1e-12)
    if not np.isfinite(embeds).all():
        raise ValueError(f"{model_type} produced non-finite normalized embeddings.")

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

    split_counts = {len(embedding) for embedding in embeddings}
    if len(split_counts) != 1:
        raise ValueError(
            f"Embedding branches have different split counts: {sorted(split_counts)}"
        )

    split_count = split_counts.pop()
    return tuple(
        np.concatenate(
            [embedding[split_index] for embedding in embeddings], axis=1
        )
        for split_index in range(split_count)
    )
