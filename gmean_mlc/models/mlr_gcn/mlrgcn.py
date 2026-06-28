import torch
import torch.nn as nn

from pdb import set_trace as breakpoint

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from copy import deepcopy
import torch.nn.functional as F

_tokenizer = _Tokenizer()

__all__ = ['mlrgcn', 'Mlrgcn']


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model_conv_proj(state_dict or model.state_dict(), cfg)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class MLCPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx_pos = cfg.TRAINER.COOP_MLC.N_CTX_POS
        n_ctx_neg = cfg.TRAINER.COOP_MLC.N_CTX_NEG
        ctx_init_pos = cfg.TRAINER.COOP_MLC.POSITIVE_PROMPT_INIT.strip()
        ctx_init_neg = cfg.TRAINER.COOP_MLC.NEGATIVE_PROMPT_INIT.strip()
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        if ctx_init_pos and ctx_init_neg:
            # use given words to initialize context vectors
            ctx_init_pos = ctx_init_pos.replace("_", " ")
            ctx_init_neg = ctx_init_neg.replace("_", " ")
            n_ctx_pos = len(ctx_init_pos.split(" "))
            n_ctx_neg = len(ctx_init_neg.split(" "))
            prompt_pos = clip.tokenize(ctx_init_pos)
            prompt_neg = clip.tokenize(ctx_init_neg)
            with torch.no_grad():
                embedding_pos = clip_model.token_embedding(prompt_pos).type(dtype)
                embedding_neg = clip_model.token_embedding(prompt_neg).type(dtype)
            ctx_vectors_pos = embedding_pos[0, 1: 1 + n_ctx_pos, :]
            ctx_vectors_neg = embedding_neg[0, 1: 1 + n_ctx_neg, :]
            prompt_prefix_pos = ctx_init_pos
            prompt_prefix_neg = ctx_init_neg
            if cfg.TRAINER.COOP_MLC.CSC:
                ctx_vectors_pos_ = []
                ctx_vectors_neg_ = []
                for _ in range(n_cls):
                    ctx_vectors_pos_.append(deepcopy(ctx_vectors_pos))
                    ctx_vectors_neg_.append(deepcopy(ctx_vectors_neg))
                ctx_vectors_pos = torch.stack(ctx_vectors_pos_, dim=0)
                ctx_vectors_neg = torch.stack(ctx_vectors_neg_, dim=0)

        else:
            # Random Initialization
            if cfg.TRAINER.COOP_MLC.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors_pos = torch.empty(n_cls, n_ctx_pos, ctx_dim, dtype=dtype)
                ctx_vectors_neg = torch.empty(n_cls, n_ctx_neg, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors_pos = torch.empty(n_ctx_pos, ctx_dim, dtype=dtype)
                ctx_vectors_neg = torch.empty(n_ctx_neg, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors_pos, std=0.02)
            nn.init.normal_(ctx_vectors_neg, std=0.02)
            prompt_prefix_pos = " ".join(["X"] * n_ctx_pos)
            prompt_prefix_neg = " ".join(["X"] * n_ctx_neg)

        print(f'Initial positive context: "{prompt_prefix_pos}"')
        print(f'Initial negative  context: "{prompt_prefix_neg}"')
        print(f"Number of positive context words (tokens): {n_ctx_pos}")
        print(f"Number of negative context words (tokens): {n_ctx_neg}")

        self.ctx_pos = nn.Parameter(ctx_vectors_pos)  # to be optimized
        self.ctx_neg = nn.Parameter(ctx_vectors_neg)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts_pos = [prompt_prefix_pos + " " + name + "." for name in classnames]
        prompts_neg = [prompt_prefix_neg + " " + name + "." for name in classnames]

        tokenized_prompts_pos = []
        tokenized_prompts_neg = []
        for p_pos, p_neg in zip(prompts_pos, prompts_neg):
            tokenized_prompts_pos.append(clip.tokenize(p_pos))
            tokenized_prompts_neg.append(clip.tokenize(p_neg))
        tokenized_prompts_pos = torch.cat(tokenized_prompts_pos)
        tokenized_prompts_neg = torch.cat(tokenized_prompts_neg)
        with torch.no_grad():
            embedding_pos = clip_model.token_embedding(tokenized_prompts_pos).type(dtype)
            embedding_neg = clip_model.token_embedding(tokenized_prompts_neg).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix_pos", embedding_pos[:, :1, :] )
        self.register_buffer("token_suffix_pos", embedding_pos[:, 1 + n_ctx_pos:, :])
        self.register_buffer("token_prefix_neg", embedding_neg[:, :1, :])
        self.register_buffer("token_suffix_neg", embedding_neg[:, 1 + n_ctx_neg:, :])

        self.n_cls = n_cls
        self.n_ctx_pos = n_ctx_pos
        self.n_ctx_neg = n_ctx_neg
        tokenized_prompts = torch.cat([tokenized_prompts_neg, tokenized_prompts_pos], dim=0)  # torch.Tensor
        self.register_buffer("tokenized_prompts", tokenized_prompts)
        self.name_lens = name_lens

    def forward(self, cls_id=None):
        ctx_pos = self.ctx_pos
        ctx_neg = self.ctx_neg

        if ctx_pos.dim() == 2:
            if cls_id is None:
                ctx_pos = ctx_pos.unsqueeze(0).expand(self.n_cls, -1, -1)
            else:
                ctx_pos = ctx_pos.unsqueeze(0).expand(len(cls_id), -1, -1)
        else:
            if cls_id is not None:
                ctx_pos = ctx_pos[cls_id]

        if ctx_neg.dim() == 2:
            if cls_id is None:
                ctx_neg = ctx_neg.unsqueeze(0).expand(self.n_cls, -1, -1)
            else:
                ctx_neg = ctx_neg.unsqueeze(0).expand(len(cls_id), -1, -1)
        else:
            if cls_id is not None:
                ctx_neg = ctx_neg[cls_id]

        if cls_id is None:
            prefix_pos = self.token_prefix_pos
            prefix_neg = self.token_prefix_neg
            suffix_pos = self.token_suffix_pos
            suffix_neg = self.token_suffix_neg
        else:
            prefix_pos = self.token_prefix_pos[cls_id]
            prefix_neg = self.token_prefix_neg[cls_id]
            suffix_pos = self.token_suffix_pos[cls_id]
            suffix_neg = self.token_suffix_neg[cls_id]


        prompts_pos = torch.cat(
            [
                prefix_pos,  # (n_cls, 1, dim)
                ctx_pos,  # (n_cls, n_ctx, dim)
                suffix_pos,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        prompts_neg = torch.cat(
            [
                prefix_neg,  # (n_cls, 1, dim)
                ctx_neg,  # (n_cls, n_ctx, dim)
                suffix_neg,  # (n_cls, *, dim)
            ],
            dim=1,
        )
        prompts = torch.cat([prompts_neg, prompts_pos], dim=0)

        if cls_id is not None:
            tokenized_prompts_pos = self.tokenized_prompts[self.n_cls:][cls_id]
            tokenized_prompts_neg = self.tokenized_prompts[:self.n_cls][cls_id]
            tokenized_prompts = torch.cat([tokenized_prompts_neg, tokenized_prompts_pos], dim=0)
        else:
            tokenized_prompts = self.tokenized_prompts


        return prompts, tokenized_prompts
    
####################### GRAPH CONVOLUTION NETWORK #######################

class GraphConvolution(nn.Module):
    
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.parameter.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.parameter.Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        if(self.weight.shape[0] == 1):
            input = input.unsqueeze(2)
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

###########################################################################

class Mlrgcn(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.visual_encoder_type = cfg.MODEL.BACKBONE.NAME
        self.prompt_learner = MLCPromptLearner(cfg, classnames, clip_model)

        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = cfg.TRAINER.COOP_MLC.LS
        self.dtype = clip_model.dtype
        self.cfg = cfg
        self.relation_path = cfg.path_to_relation
        ########################################################################### 

        if cfg.DATASET.NAME == 'unimib':
            self.gc1 = GraphConvolution(1, 4)
            self.gc2 = GraphConvolution(4, 4)
            self.gc3 = GraphConvolution(4, 1)
            self.relu = nn.LeakyReLU(0.2)
            self.relu2 = nn.LeakyReLU(0.2)
        
        elif cfg.DATASET.NAME == 'voc2007':
            self.gc1 = GraphConvolution(1, 10)
            self.gc2 = GraphConvolution(10, 20)
            self.gc3 = GraphConvolution(20, 1)
            self.relu = nn.LeakyReLU(0.2)
            self.relu2 = nn.LeakyReLU(0.2)
        
        elif cfg.DATASET.NAME=='foodseg103':
            self.gc1 = GraphConvolution(1, 5)
            self.gc2 = GraphConvolution(5, 10)
            self.gc3 = GraphConvolution(10, 1)
            self.relu = nn.LeakyReLU(0.2)
            self.relu2 = nn.LeakyReLU(0.2)


        else:
            self.gc1 = GraphConvolution(1, 10)
            self.gc2 = GraphConvolution(10, 10)
            self.gc3 = GraphConvolution(10, 1)
            # self.gc4 = GraphConvolution(200, 25)
            # self.gc5 = GraphConvolution(25, 1)
            # self.gc6 = GraphConvolution(200, 25)
            # self.gc7 = GraphConvolution(25, 1)
            # self.gc2 = GraphConvolution(10, 20)
            # self.gc3 = GraphConvolution(20, 1)
            self.relu = nn.LeakyReLU(0.2)
            self.relu2 = nn.LeakyReLU(0.2)


        if cfg.DATASET.NAME == 'unimib':
            self.relation = torch.load(self.relation_path)

            _ ,max_idx = torch.topk(self.relation, len(self.relation))
            mask = torch.ones_like(self.relation).type(torch.bool)
            for i, idx in enumerate(max_idx):
                mask[i][idx] = 0
            self.relation[mask] = 0
            
            sparse_mask = mask
            dialog = torch.eye(len(self.relation)).type(torch.bool)

            sum_relation = torch.sum(self.relation, dim=1)
            zero_mask = (sum_relation == 0)
            sum_relation = torch.where(zero_mask, torch.tensor(1e-4), sum_relation)
            self.relation = self.relation / sum_relation.reshape(-1, 1) * 0.2

        elif cfg.DATASET.NAME == 'foodseg103':
            self.relation = torch.load(self.relation_path)

            _ ,max_idx = torch.topk(self.relation, len(self.relation))
            mask = torch.ones_like(self.relation).type(torch.bool)
            for i, idx in enumerate(max_idx):
                mask[i][idx] = 0
            self.relation[mask] = 0
            
            sparse_mask = mask
            dialog = torch.eye(len(self.relation)).type(torch.bool)
            self.relation[dialog] = 0
            # breakpoint()
            self.relation = self.relation / torch.sum(self.relation, dim=1).reshape(-1, 1) * 0.2
            self.relation[dialog] = 1-0.2

            self.relation = self.relation/torch.diag(self.relation)
        
        elif 'voc' in cfg.DATASET.NAME:
            self.relation = torch.load(self.relation_path)

            _ ,max_idx = torch.topk(self.relation, len(self.relation))
            mask = torch.ones_like(self.relation).type(torch.bool)
            for i, idx in enumerate(max_idx):
                mask[i][idx] = 0
            self.relation[mask] = 0
            
            sparse_mask = mask
            dialog = torch.eye(len(self.relation)).type(torch.bool)
            self.relation[dialog] = 0
            # breakpoint()
            self.relation = self.relation / torch.sum(self.relation, dim=1).reshape(-1, 1) * 0.2
            self.relation[dialog] = 1-0.2

            self.relation = self.relation/torch.diag(self.relation)

        elif 'coco' in cfg.DATASET.NAME:
            self.relation = torch.load(self.relation_path)

            _ ,max_idx = torch.topk(self.relation, len(self.relation))
            mask = torch.ones_like(self.relation).type(torch.bool)
            for i, idx in enumerate(max_idx):
                mask[i][idx] = 0
            self.relation[mask] = 0
            
            sparse_mask = mask
            dialog = torch.eye(len(self.relation)).type(torch.bool)
            self.relation[dialog] = 0
            # breakpoint()
            self.relation = self.relation / torch.sum(self.relation, dim=1).reshape(-1, 1) * 0.2
            self.relation[dialog] = 1-0.2

            self.relation = self.relation/torch.diag(self.relation)

        self.relation[dialog] = 1-0.2
        
        self.gcn_relation = self.relation.clone()
        if torch.any(torch.isnan(self.gcn_relation)):
            print("self.gcn_relation contains NaN values")
            breakpoint()
        assert(self.gcn_relation.requires_grad == False)

        self.relation = torch.exp(self.relation/0.2) / torch.sum(torch.exp(self.relation/0.2), dim=1).reshape(-1,1)
        self.relation[sparse_mask] = 0
        self.relation = self.relation / torch.sum(self.relation, dim=1).reshape(-1, 1)
        
        ###########################################################################

    def forward(self, image, cls_id=None):
        # get image and text features

        image_features, attn_weights = self.image_encoder(image.type(self.dtype))
        prompts, tokenized_prompts = self.prompt_learner(cls_id)
        text_features = self.text_encoder(prompts, tokenized_prompts)
        

        # normalize features
        image_features_norm = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # print('text_features', text_features.shape)
        ######################################################
        # Class-Specific Region Feature Aggregation
        output = 20 * F.conv1d(image_features_norm, text_features[:, :, None])
        
        b, c, _ = output.shape
        output_half = output[:,  c // 2:]
        w_half = F.softmax(output_half, dim=-1)
        w = torch.cat([w_half, w_half], dim=1)
        output = 5 * (output * w).sum(-1)

        b, c = output.shape
        # resized_output = output.resize(b, 2, c//2)
        resized_output_neg = output[:,:c//2]#.unsqueeze(-1) ## 32,103 
        resized_output_pos = output[:,c//2:]#.unsqueeze(-1)
        
        
        identity = output
        
        tf1 = resized_output_neg     
        # if torch.any(torch.isnan(self.gcn_relation)):
        #     print("gcn_relation contains NaN values")
        #     breakpoint()   
        tf1 = self.gc1(tf1, self.gcn_relation.cuda())
        # if torch.any(torch.isnan(tf1)):
        #     print("tf1 contains NaN values")
        #     breakpoint()
        tf1 = self.relu(tf1)
        tf1 = self.gc2(tf1, self.gcn_relation.cuda())
        tf1 = self.relu2(tf1)
        tf1 = self.gc3(tf1, self.gcn_relation.cuda())
        # tf1 = self.relu2(tf1)
        # tf1 = self.gc4(tf1, self.gcn_relation.cuda())
        # tf1 = self.relu2(tf1)
        # tf1 = self.gc5(tf1, self.gcn_relation.cuda())
        # tf1 = self.relu2(tf1)
        # tf1 = self.gc6(tf1, self.gcn_relation.cuda())
        # tf1 = self.relu2(tf1)
        # tf1 = self.gc7(tf1, self.gcn_relation.cuda())

        tf2 = resized_output_pos        
        tf2 = self.gc1(tf2, self.gcn_relation.cuda())
        tf2 = self.relu(tf2)
        tf2 = self.gc2(tf2, self.gcn_relation.cuda())
        tf2 = self.relu2(tf2)
        tf2 = self.gc3(tf2, self.gcn_relation.cuda())
        # tf2 = self.relu2(tf2)
        # tf2 = self.gc4(tf2, self.gcn_relation.cuda())
        # tf2 = self.relu2(tf2)
        # tf2 = self.gc5(tf2, self.gcn_relation.cuda())
        # tf2 = self.relu2(tf2)
        # tf2 = self.gc6(tf2, self.gcn_relation.cuda())
        # tf2 = self.relu2(tf2)
        # tf2 = self.gc7(tf2, self.gcn_relation.cuda())

        output = torch.cat((tf1,tf2), dim = 1).squeeze(-1)
        output += identity
        
        # convert the shape of logits to [b, 2, num_class]
        logits = output.resize(b, 2, c//2)
        return logits

    @property
    def network_name(self):
        name = ''
        name += 'Mlrgcn-{}'.format(self.visual_encoder_type)
        return name

    def backbone_params(self):
        params = []
        for name, param in self.named_parameters():
            if "image_encoder" in name and "prompt_learner" not in name and 'attnpool' not in name:
                params.append(param)
        return params

    def attn_params(self):
        params = []
        for name, param in self.named_parameters():
            if 'attnpool' in name and 'image_encoder' in name:
                params.append(param)
                # print(name)
        return params

    def prompt_params(self):
        params = []
        for name, param in self.named_parameters():
            if "prompt_learner" in name:
                params.append(param)
        return params

def mlrgcn(cfg, classnames, **kwargs):
    print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
    clip_model = load_clip_to_cpu(cfg)

    clip_model.float()

    print("Building mlrgcn")
    model = Mlrgcn(cfg, classnames, clip_model)
    if not cfg.TRAINER.FINETUNE_BACKBONE:
        print('Freeze the backbone weights')
        backbone_params = model.backbone_params()
        for param in backbone_params:
            param.requires_grad_(False)

    if not cfg.TRAINER.FINETUNE_ATTN:
        print('Freeze the attn weights')
        attn_params = model.attn_params()
        for param in attn_params:
            param.requires_grad_(False)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)

    # Note that multi-gpu training could be slow because CLIP's size is
    # big, which slows down the copy operation in DataParallel
    device_count = torch.cuda.device_count()
    if device_count > 1:
        print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        model = nn.DataParallel(model)
    return model