import math
import torch
from torch import nn
import torch.nn.functional as F

class HINResidualBlock(nn.Module):
    def __init__(self, cin, cout, time_dim=None, dropout=.1):
        super().__init__()
        if cout % 2: raise ValueError('HIN channels must be even')
        self.conv1=nn.Conv2d(cin,cout,3,padding=1)
        self.norm=nn.InstanceNorm2d(cout//2,affine=True,eps=1e-5)
        self.time=nn.Linear(time_dim,cout) if time_dim else None
        self.conv2=nn.Conv2d(cout,cout,3,padding=1)
        self.drop=nn.Dropout(dropout)
        self.shortcut=nn.Identity() if cin==cout else nn.Conv2d(cin,cout,1)
    def forward(self,x,time=None):
        h=self.conv1(F.silu(x)); a,b=h.chunk(2,1)
        h=torch.cat((self.norm(a),b),1)
        if self.time is not None: h=h+self.time(F.silu(time))[:,:,None,None]
        return self.shortcut(x)+self.conv2(self.drop(F.silu(h)))

class Encoder(nn.Module):
    def __init__(self, cin, widths, blocks, time_dim, dropout):
        super().__init__()
        self.input=nn.Conv2d(cin,widths[0],3,padding=1)
        self.layers=nn.ModuleList(); self.skip_channels=[widths[0]]
        cur=widths[0]
        for level,width in enumerate(widths):
            for _ in range(blocks):
                self.layers.append(HINResidualBlock(cur,width,time_dim,dropout)); cur=width
                self.skip_channels.append(cur)
            if level<len(widths)-1:
                self.layers.append(nn.Conv2d(cur,cur,3,stride=2,padding=1))
                self.skip_channels.append(cur)
    def forward(self,x,time=None,addition=None):
        h=self.input(x)
        if addition is not None: h=h+addition
        skips=[h]
        for layer in self.layers:
            h=layer(h,time) if isinstance(layer,HINResidualBlock) else layer(h)
            skips.append(h)
        return h,skips

class Middle(nn.Module):
    def __init__(self,width,dropout):
        super().__init__()
        self.pre=HINResidualBlock(width,width,None,dropout)
        self.dilated=nn.ModuleList([nn.Conv2d(width,width,3,padding=d,dilation=d) for d in (2,4,8,16)])
        self.post=HINResidualBlock(width,width,None,dropout)
    def forward(self,x):
        h=self.pre(x)
        for conv in self.dilated: h=conv(F.silu(h))
        return self.post(h)

class UNet(nn.Module):
    def __init__(self, base=32, mult=(1,2,3,4), blocks=1, dropout=.1):
        super().__init__()
        widths=[base*m for m in mult]
        self.base=base
        self.time=nn.Sequential(nn.Linear(base,128),nn.SiLU(),nn.Linear(128,128))
        self.encoder=Encoder(12,widths,blocks,128,dropout)
        self.middle=Middle(widths[-1],dropout)
        stack=list(self.encoder.skip_channels); cur=widths[-1]
        self.decoder=nn.ModuleList()
        for level in reversed(range(len(widths))):
            for _ in range(blocks+1):
                self.decoder.append(HINResidualBlock(cur+stack.pop(),widths[level],128,dropout)); cur=widths[level]
            if level: self.decoder.append(nn.ConvTranspose2d(cur,cur,4,stride=2,padding=1))
        assert not stack
        self.output=nn.Conv2d(cur,3,3,padding=1)
    def forward(self,state,index,condition):
        freq=torch.exp(torch.arange(self.base//2,device=state.device)*(-math.log(10000)/(self.base//2-1)))
        phase=index.float()[:,None]*freq[None]
        time=self.time(torch.cat((phase.sin(),phase.cos()),1))
        x=torch.cat((state,condition['image'],condition['physical'],condition['histogram']),1)
        h,skips=self.encoder(x,time,condition['highfreq'])
        h=self.middle(h)
        for layer in self.decoder:
            if isinstance(layer,HINResidualBlock): h=layer(torch.cat((h,skips.pop()),1),time)
            else: h=layer(h)
        assert not skips
        return self.output(F.silu(h))

class BetaUNet(nn.Module):
    def __init__(self, base=32, mult=(1,2,3,4), blocks=1, dropout=.1):
        super().__init__()
        widths=[base*m for m in mult]
        self.encoder=Encoder(3,widths,blocks,None,dropout)
        self.middle=Middle(widths[-1],dropout)
        self.projection=nn.Identity() if widths[-1]==128 else nn.Conv2d(widths[-1],128,1)
        self.head=nn.Sequential(nn.Linear(128,3),nn.SiLU(),nn.Linear(3,3),nn.Sigmoid())
    def forward(self,x):
        h,_=self.encoder(x)
        h=self.projection(self.middle(h)).mean((2,3))
        return self.head(h).view(x.shape[0],3,1,1)
