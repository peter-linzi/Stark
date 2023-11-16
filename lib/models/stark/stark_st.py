"""
STARK-ST Model (Spatio-Temporal).
"""
from lib.utils.timer import Timer
from .backbone import build_backbone
from .transformer import build_transformer
from .head import build_box_head, MLP
from lib.models.stark.stark_s import STARKS
from lib.utils.timer import Timer
import torch
from lib.utils.box_ops import box_xyxy_to_cxcywh

class STARKST(STARKS):
    """ This is the base class for Transformer Tracking """
    def __init__(self, backbone, transformer, box_head, num_queries,
                 aux_loss=False, head_type="CORNER", cls_head=None):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__(backbone, transformer, box_head, num_queries,
                         aux_loss=aux_loss, head_type=head_type)
        self.cls_head = cls_head
        self.t_head = Timer()
        
    # def forward(self, img=None, seq_dict=None, mode="backbone", run_box_head=False, run_cls_head=False):
    #     if mode == "backbone":
    #         return self.forward_backbone(img)
    #     elif mode == "transformer":
    #         return self.forward_transformer(seq_dict, run_box_head=run_box_head, run_cls_head=run_cls_head)
    #     else:
    #         raise ValueError

    # forward backbone
    def forward(self, tensors: torch.Tensor, mask:torch.Tensor):
        """The input type is NestedTensor, which consists of:
                - tensor: batched images, of shape [batch_size x 3 x H x W]
                - mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        # assert isinstance(input, NestedTensor)
        # Forward the backbone
        tensor, mask, pos = self.backbone(tensors, mask)  # features & masks, position embedding for the search
        # Adjust the shapes
        return self.adjust(tensor, mask, pos)

    # forward transformer + head
    # def forward(self, feat, mask, pos_embed):
    #     if self.aux_loss:
    #         raise ValueError("Deep supervision is not supported.")
    #     # Forward the transformer encoder and decoder
    #     # torch.onnx.export(self.transformer, (seq_dict["feat"], seq_dict["mask"], self.query_embed.weight, seq_dict["pos"]), "checkpoints/onnx/transformer.onnx", opset_version=11)
    #     # output_embed, enc_mem = self.transformer(feat, mask, self.query_embed.weight, pos_embed, return_encoder_output=True)
    #     output_embed, enc_mem = self.transformer(feat, mask, pos_embed, return_encoder_output=True)
    #     # Forward the corner head
    #     # self.t_head.tic()
    #     out, outputs_coord = self.forward_head(output_embed, enc_mem, run_box_head=True, run_cls_head=True)
    #     # self.t_head.toc()
    #     # print("t_head = {:.3f}".format(self.t_head.average_time))
    #     return out["pred_logits"], out["pred_boxes"]

    # forward_box_head
    # def forward(self, hs, memory):
    #     """
    #     hs: output embeddings (1, B, N, C)
    #     memory: encoder embeddings (HW1+HW2, B, C)"""
    #     if self.head_type == "CORNER":
    #         # adjust shape
    #         enc_opt = memory[-self.feat_len_s:].transpose(0, 1)  # encoder output for the search region (B, HW, C)
    #         dec_opt = hs.squeeze(0).transpose(1, 2)  # (B, C, N)
    #         att = torch.matmul(enc_opt, dec_opt)  # (B, HW, N)
    #         opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute((0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
    #         bs, Nq, C, HW = opt.size()
    #         opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)
    #         # run the corner head
    #         outputs_coord = box_xyxy_to_cxcywh(self.box_head(opt_feat))
    #         outputs_coord_new = outputs_coord.view(bs, Nq, 4)
    #         # out = {'pred_boxes': outputs_coord_new}
    #         return outputs_coord_new


    def forward_transformer(self, seq_dict, run_box_head=False, run_cls_head=False):
        if self.aux_loss:
            raise ValueError("Deep supervision is not supported.")
        # Forward the transformer encoder and decoder
        # torch.onnx.export(self.transformer, (seq_dict["feat"], seq_dict["mask"], seq_dict["pos"]), "checkpoints/onnx/transformer.onnx", opset_version=11)
        # torch.onnx.export(self.transformer, (seq_dict["feat"], seq_dict["mask"], self.query_embed.weight, seq_dict["pos"]), "checkpoints/onnx/transformer.onnx", opset_version=11)
       
        # output_embed, enc_mem = self.transformer(seq_dict["feat"], seq_dict["mask"], self.query_embed.weight,
        #                                          seq_dict["pos"], return_encoder_output=True)
        output_embed, enc_mem = self.transformer(seq_dict["feat"], seq_dict["mask"], 
                                                 seq_dict["pos"], return_encoder_output=True)
        # Forward the corner head
        self.t_head.tic()
        out, outputs_coord = self.forward_head(output_embed, enc_mem, run_box_head=run_box_head, run_cls_head=run_cls_head)
        self.t_head.toc()
        # print("t_head = {:.3f}".format(self.t_head.average_time))
        return out, outputs_coord, output_embed

    def forward_head(self, hs, memory, run_box_head=False, run_cls_head=False):
        """
        hs: output embeddings (1, B, N, C)
        memory: encoder embeddings (HW1+HW2, B, C)"""
        out_dict = {}
        if run_cls_head:
            # forward the classification head
            # torch.onnx.export(self.cls_head, hs, "checkpoints/onnx/cls_head.onnx", opset_version=11)
            out_dict.update({'pred_logits': self.cls_head(hs)[-1]})
        if run_box_head:
            # forward the box prediction head
            # torch.onnx.export(self, (hs, memory), "checkpoints/onnx/box_head.onnx", opset_version=11)
            out_dict_box, outputs_coord = self.forward_box_head(hs, memory)
            # merge results
            out_dict.update(out_dict_box)
            return out_dict, outputs_coord
        else:
            return out_dict, None


def build_starkst(cfg):
    backbone = build_backbone(cfg)  # backbone and positional encoding are built together
    transformer = build_transformer(cfg)
    box_head = build_box_head(cfg)
    cls_head = MLP(cfg.MODEL.HIDDEN_DIM, cfg.MODEL.HIDDEN_DIM, 1, cfg.MODEL.NLAYER_HEAD)
    model = STARKST(
        backbone,
        transformer,
        box_head,
        num_queries=cfg.MODEL.NUM_OBJECT_QUERIES,
        aux_loss=cfg.TRAIN.DEEP_SUPERVISION,
        head_type=cfg.MODEL.HEAD_TYPE,
        cls_head=cls_head
    )

    return model
