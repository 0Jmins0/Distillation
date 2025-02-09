class SAM2CLIP(nn.model):
    def __init__(self, sam_encoder, clip_encoder,adapter):
        super(SAM2CLIP, self).__init__()
        self.sam_encoder = sam_encoder
        self.clip_encoder = clip_encoder
        self.adapter = adapter
    
    def forward(self, x):
        sam_cls_token = self.