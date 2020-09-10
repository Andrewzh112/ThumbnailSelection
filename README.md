# Automatic Thumbnail Selection

## Objective

An automatic thumbnail selection model that picks out the best frame based on context and aesthetic.

## Data

Data consist of videos downloaded from **YouTube** and will include each video's title, description, video, and thumbnail.

## Model and Training Details

The model components includes encoders for 3 modalities: *text*, *frames* and *audio*. The modalities are aggregated using a **transformer-isque module**. With modality attention, the final output is projected to a latent. During **training**, top selections of frames with best *niqe* scores are encoded to latents and are compared and contrasted with the video latent. Here, I employed **triplet-isque loss** which the video latent will be the **anchor**, the ground truth thumbnail latent will be the **positive pair** and all other topk *niqe* scored frames will be the **negative pairs** (they will probably be semi-hard negatives since they are already topk in *niqe* scores). During **inference**, topk *niqe* scored frames will be contrasted with the video latent and the *closest* frame will be selected as the thumbnail.