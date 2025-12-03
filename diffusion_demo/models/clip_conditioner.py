import torch 
import torch.nn as nn
import open_clip

class CLIPTextConditioner(nn.Module):
    """
    CLIP -> text encoder to get text embeddings for conditioning
    - input: text_tokens [B, L]
    - output: text_embeddings [B, L, cond_dim]
    """

    def __init__(self, model_name="ViT-B-32", pretrained="openai", device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), normalize: bool = True):
        super(CLIPTextConditioner, self).__init__()
        self.device = device
        self.normalize = normalize  # assuming cond_dim=512

        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
        self.clip_model = model 
        self.tokenizer = open_clip.get_tokenizer(model_name)

        # freeze CLIP model parameters
        # usually we don't fine-tune CLIP in diffusion training
        for p in self.clip_model.parameters():
            p.requires_grad = False

        # note: cond_dim depends on the CLIP model used, con_dim of Unet should match this
        with torch.no_grad():
            dummy = self.tokenizer(["text"]).to(device)
            emb = self.clip_model.encode_text(dummy)
            self.cond_dim = emb.shape[-1]

    @torch.no_grad()
    def encode_text(self, prompts: list) -> torch.Tensor:
        """
        prompts: list of strings, len = B
        return: text_embeddings [B, L, cond_dim]
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        
        text_tokens = self.tokenizer(prompts).to(self.device)  # [B, L]
        text_embed = self.clip_model.encode_text(text_tokens)  # [B, cond_dim

        if self.normalize:
            text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)

        # in here, we return the pooled embedding as a single token per prompt
        # for more advanced usage, one can modify this to return per-token embeddings
        cond_tokens = text_embed.unsqueeze(1)  # [B, 1, cond_dim]
        return cond_tokens
    
    def forward(self, prompts: list) -> torch.Tensor:
        return self.encode_text(prompts)