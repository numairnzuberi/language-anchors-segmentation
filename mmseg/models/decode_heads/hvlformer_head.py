import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.models.builder import HEADS

from .tqdm_head import tqdmHead


class HierarchicalTextualQueryGenerator(nn.Module):
    """Generate E scale-aligned textual queries from class text embeddings."""

    def __init__(self, text_dim, query_dim, num_levels=3, prompt_tokens=8):
        super().__init__()
        self.num_levels = num_levels
        self.prompt_tokens = prompt_tokens
        self.level_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(text_dim, query_dim),
                nn.ReLU(inplace=True),
                nn.Linear(query_dim, query_dim))
            for _ in range(num_levels)
        ])
        self.relevance_mlp = nn.Sequential(
            nn.Linear(text_dim + query_dim, query_dim),
            nn.ReLU(inplace=True),
            nn.Linear(query_dim, 1))

    def forward(self, texts, image_context):
        """Return hierarchical queries and the diversity regularizer.

        Args:
            texts (Tensor): Class text embeddings with shape [B, K, C].
            image_context (Tensor): Pooled visual context with shape [B, D].
        """
        level_queries = [head(texts) for head in self.level_heads]
        queries = torch.stack(level_queries, dim=1)
        context = image_context.unsqueeze(1).expand(-1, texts.size(1), -1)
        scores = torch.sigmoid(self.relevance_mlp(torch.cat([texts, context], dim=-1)))
        queries = queries * scores.unsqueeze(1)
        return queries, self.diversity_loss(queries)

    @staticmethod
    def diversity_loss(queries):
        if queries.size(1) < 2:
            return queries.new_tensor(0.)
        q = F.normalize(queries, dim=-1)
        sim = torch.einsum('bekd,bfkd->bekf', q, q)
        eye = torch.eye(q.size(1), device=q.device, dtype=torch.bool)
        return sim.masked_select(~eye.view(1, q.size(1), 1, q.size(1))).pow(2).mean()


class PixelTextRefinementModule(nn.Module):
    """Spatially gate text and pixel features before masked decoding."""

    def __init__(self, dim):
        super().__init__()
        self.text_proj = nn.Linear(dim, dim)
        self.visual_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.attn = nn.Sequential(
            nn.Conv2d(dim * 3, dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, 3, kernel_size=1),
            nn.Sigmoid())
        self.visual_out = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, queries, pixel_feature):
        b, k, c = queries.shape
        h, w = pixel_feature.shape[-2:]
        text_map = self.text_proj(queries).mean(1).view(b, c, 1, 1).expand(-1, -1, h, w)
        visual_map = self.visual_proj(pixel_feature)
        fused = text_map + visual_map
        gates = self.attn(torch.cat([text_map, visual_map, fused], dim=1))
        text_gate, visual_gate, fused_gate = gates.chunk(3, dim=1)
        refined_visual = self.visual_out(visual_map * text_gate * fused_gate)
        query_context = (text_map * visual_gate * fused_gate).flatten(2).mean(-1)
        refined_queries = queries + query_context.unsqueeze(1)
        return refined_queries, refined_visual, gates


@HEADS.register_module()
class HVLFormerHead(tqdmHead):
    """TQDM head extended with HTQG, PTRM, and CMCR-ready outputs."""

    def __init__(self,
                 hvl_num_levels=3,
                 hvl_prompt_tokens=8,
                 lambda_div=1.0,
                 lambda_cmcr=5.0,
                 **kwargs):
        super().__init__(**kwargs)
        text_dim = kwargs['text_proj'].text_out_dim
        self.lambda_div = lambda_div
        self.lambda_cmcr = lambda_cmcr
        self.htqg = HierarchicalTextualQueryGenerator(
            text_dim=text_dim,
            query_dim=self.decoder_embed_dims,
            num_levels=hvl_num_levels,
            prompt_tokens=hvl_prompt_tokens)
        self.ptrm = nn.ModuleList([
            PixelTextRefinementModule(self.decoder_embed_dims)
            for _ in range(hvl_num_levels)
        ])

    def build_hierarchical_queries(self, texts, multi_scale_memorys):
        image_context = multi_scale_memorys[0].flatten(2).mean(-1)
        projected_text = self.text_proj(texts)
        hierarchical_queries, loss_div = self.htqg(projected_text, image_context)
        refined_memorys = []
        refined_queries = []
        attention_maps = []
        for level_idx, memory in enumerate(multi_scale_memorys):
            queries_l, memory_l, gates_l = self.ptrm[level_idx](
                hierarchical_queries[:, level_idx], memory)
            refined_queries.append(queries_l)
            refined_memorys.append(memory_l)
            attention_maps.append(gates_l)
        query_feat = torch.stack(refined_queries, dim=1).flatten(1, 2).permute(1, 0, 2)
        return query_feat, refined_memorys, attention_maps, loss_div

    def forward(self, feats, texts, img_metas, return_mask_features=False,
                get_similarity=False, return_hvl=False):
        mask_features, multi_scale_memorys = self.pixel_decoder(feats, texts)
        query_feat, multi_scale_memorys, attention_maps, loss_div = \
            self.build_hierarchical_queries(texts, multi_scale_memorys)

        batch_size = len(img_metas)
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            decoder_input = decoder_input.flatten(2).permute(2, 0, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            mask = decoder_input.new_zeros(
                (batch_size,) + multi_scale_memorys[i].shape[-2:], dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(2).permute(2, 0, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)

        query_embed = self.query_embed.weight.unsqueeze(1).repeat((1, batch_size, 1))
        if query_embed.size(0) != query_feat.size(0):
            query_embed = query_embed[:query_feat.size(0)]

        cls_pred_list, mask_pred_list = [], []
        cls_pred, mask_pred, attn_mask = self.forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:],
            get_similarity=get_similarity)
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            layer = self.transformer_decoder.layers[i]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                attn_masks=[attn_mask, None],
                query_key_padding_mask=None,
                key_padding_mask=None)
            cls_pred, mask_pred, attn_mask = self.forward_head(
                query_feat, mask_features,
                multi_scale_memorys[(i + 1) % self.num_transformer_feat_level].shape[-2:],
                get_similarity=get_similarity)
            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)

        if return_hvl:
            return cls_pred_list, mask_pred_list, dict(
                loss_div=loss_div, attention_maps=attention_maps)
        if return_mask_features:
            return cls_pred_list, mask_pred_list, mask_features
        return cls_pred_list, mask_pred_list

    def forward_train(self, x, texts, img_metas, gt_semantic_seg, train_cfg,
                      gt_labels, gt_masks):
        all_cls_scores, all_mask_preds, hvl_state = self(
            x, texts, img_metas, return_hvl=True)
        losses = self.loss(all_cls_scores, all_mask_preds, gt_labels, gt_masks,
                           img_metas)
        losses['loss_div'] = hvl_state['loss_div'] * self.lambda_div
        return losses
