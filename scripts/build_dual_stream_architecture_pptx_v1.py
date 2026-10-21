"""Draw editable OOXML architecture slides, retaining the source slide as reference."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile
import xml.etree.ElementTree as E

NS={'p':'http://schemas.openxmlformats.org/presentationml/2006/main',
    'a':'http://schemas.openxmlformats.org/drawingml/2006/main',
    'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
for prefix,uri in NS.items(): E.register_namespace(prefix,uri)
EMU=914400
W,H=18,10.125
INK='20364B';GRAY='647487';REC='C77933';SEM='4D8062';BLUE='4477A4'


def tag(name):
    prefix,local=name.split(':');return '{'+NS[prefix]+'}'+local


def el(parent,element_name,attrs=None,**kwargs):
    return E.SubElement(parent,tag(element_name),{k:str(v) for k,v in (attrs or {}).items()}|{k:str(v) for k,v in kwargs.items()})


def units(v): return str(round(v*EMU))


class Slide:
    def __init__(self):
        self.root=E.Element(tag('p:sld'))
        cs=el(self.root,'p:cSld');self.tree=el(cs,'p:spTree')
        nv=el(self.tree,'p:nvGrpSpPr');el(nv,'p:cNvPr',id=1,name='');el(nv,'p:cNvGrpSpPr');el(nv,'p:nvPr')
        tr=el(el(self.tree,'p:grpSpPr'),'a:xfrm')
        for name,attr in [('off',dict(x=0,y=0)),('ext',dict(cx=0,cy=0)),('chOff',dict(x=0,y=0)),('chExt',dict(cx=0,cy=0))]:el(tr,'a:'+name,attr)
        el(el(self.root,'p:clrMapOvr'),'a:masterClrMapping')
        self.idx=2

    def ident(self):
        idx=self.idx;self.idx+=1;return idx

    def shape(self,x,y,w,h,text='',fill='FFFFFF',line=None,size=14,color=INK,
              bold=False,round_=True,dash=False,name=None,align='ctr',margins=.06):
        idx=self.ident();sp=el(self.tree,'p:sp');nv=el(sp,'p:nvSpPr')
        el(nv,'p:cNvPr',id=idx,name=name or text.replace('\n',' ')[:70]);el(nv,'p:cNvSpPr');el(nv,'p:nvPr')
        pr=el(sp,'p:spPr');tr=el(pr,'a:xfrm');el(tr,'a:off',x=units(x),y=units(y));el(tr,'a:ext',cx=units(w),cy=units(h))
        el(el(pr,'a:prstGeom',prst='roundRect' if round_ else 'rect'),'a:avLst')
        if fill:el(el(pr,'a:solidFill'),'a:srgbClr',val=fill)
        else:el(pr,'a:noFill')
        ln=el(pr,'a:ln',w=12700)
        if line:el(el(ln,'a:solidFill'),'a:srgbClr',val=line)
        else:el(ln,'a:noFill')
        if dash:el(ln,'a:prstDash',val='dash')
        if text:
            tx=el(sp,'p:txBody');el(tx,'a:bodyPr',wrap='square',anchor='ctr',lIns=units(margins),rIns=units(margins),tIns=units(margins),bIns=units(margins));el(tx,'a:lstStyle')
            for textline in text.split('\n'):
                p=el(tx,'a:p');pp=el(p,'a:pPr',algn=align)
                el(el(pp,'a:spcAft'),'a:spcPts',val=0)
                r=el(p,'a:r');rp=el(r,'a:rPr',lang='zh-CN',sz=round(size*100),b=int(bold))
                el(el(rp,'a:solidFill'),'a:srgbClr',val=color)
                el(rp,'a:latin',typeface='Arial');el(rp,'a:ea',typeface='Noto Sans CJK SC')
                el(r,'a:t').text=textline
                el(p,'a:endParaRPr',lang='zh-CN',sz=round(size*100))
        return idx

    def text(self,x,y,w,h,text,size=14,color=INK,bold=False,align='ctr'):
        return self.shape(x,y,w,h,text,fill=None,size=size,color=color,bold=bold,round_=False,align=align,margins=.01)

    def line(self,points,color=INK,dash=False,arrow=True,width=1.5,name='Connector'):
        idx=self.ident();sp=el(self.tree,'p:sp');nv=el(sp,'p:nvSpPr')
        el(nv,'p:cNvPr',id=idx,name=name);el(nv,'p:cNvSpPr');el(nv,'p:nvPr')
        pr=el(sp,'p:spPr');x0=min(p[0] for p in points);y0=min(p[1] for p in points)
        w=max(max(p[0] for p in points)-x0,.001);h=max(max(p[1] for p in points)-y0,.001)
        tr=el(pr,'a:xfrm');el(tr,'a:off',x=units(x0),y=units(y0));el(tr,'a:ext',cx=units(w),cy=units(h))
        geom=el(pr,'a:custGeom')
        for n in ('avLst','gdLst','ahLst','cxnLst'):el(geom,'a:'+n)
        el(geom,'a:rect',l=0,t=0,r='r',b='b');path=el(el(geom,'a:pathLst'),'a:path',w=units(w),h=units(h),fill='none')
        for i,(x,y) in enumerate(points):el(el(path,'a:moveTo' if i==0 else 'a:lnTo'),'a:pt',x=units(x-x0),y=units(y-y0))
        el(pr,'a:noFill');ln=el(pr,'a:ln',w=round(width*12700));el(el(ln,'a:solidFill'),'a:srgbClr',val=color)
        if dash:el(ln,'a:prstDash',val='dash')
        if arrow:el(ln,'a:tailEnd',type='triangle',w='sm',len='sm')
        return idx

    def pic(self,rid,x,y,w,h,name):
        sp=el(self.tree,'p:pic');nv=el(sp,'p:nvPicPr');el(nv,'p:cNvPr',id=self.ident(),name=name)
        el(el(nv,'p:cNvPicPr'),'a:picLocks',noChangeAspect=1);el(nv,'p:nvPr')
        bf=el(sp,'p:blipFill');el(bf,'a:blip',{tag('r:embed'):rid});el(el(bf,'a:stretch'),'a:fillRect')
        pr=el(sp,'p:spPr');tr=el(pr,'a:xfrm');el(tr,'a:off',x=units(x),y=units(y));el(tr,'a:ext',cx=units(w),cy=units(h))
        el(el(pr,'a:prstGeom',prst='rect'),'a:avLst')

    def tokens(self,x,y,vertical,color,name):
        for i in range(3):
            xx=x if vertical else x+i*.35;yy=y+i*.34 if vertical else y
            self.shape(xx,yy,.28,.26,fill=color,line=None,name=f'{name} token {i+1}')
            for j in range(3):self.line([(xx+.06,yy+.065+j*.06),(xx+.22,yy+.065+j*.06)],color='FFFFFF',arrow=False,width=.7,name=f'{name} editable token mark')

    def title(self,title,subtitle,page):
        self.text(.45,.20,17.1,.55,title,26,bold=True,align='l')
        self.text(.46,.83,17,.42,subtitle,11.5,color=GRAY,align='l')
        self.line([(.45,9.47),(17.55,9.47)],color='D9E0E6',arrow=False,width=.8)
        self.text(.47,9.57,15.9,.31,'设计草案｜实线：主路径；虚线：待定模块或可选写回｜不表示已实现、已验证的性能',10,color=GRAY,align='l')
        self.text(16.7,9.57,.8,.3,f'{page} / 3',11,color=GRAY)

    def xml(self):return E.tostring(self.root,encoding='UTF-8',xml_declaration=True)


def architecture():
    s=Slide();s.title('几何–语义双流 Gaussian 架构（建议改版）',
        '保留两组token；上层负责完整重建，下层负责对象实例。SAM与语义特征来源仍可替换。',1)
    # Backgrounds and major modules; arrows remain separate editable objects.
    s.shape(7.05,3.42,5.82,5.60,fill='F5F6F8',line='D7DDE3',name='Alternating stages container')
    s.text(7.25,3.48,5.4,.34,'交替交互阶段 × L',13,color=GRAY)
    s.text(.35,2.80,1.35,.35,'输入RGB',13,bold=True)
    s.pic('rId2',.48,3.28,1.08,1.08,'Input context RGB 1')
    s.pic('rId3',.48,4.60,1.08,1.08,'Input context RGB 2')
    s.shape(2.05,1.80,2.18,.84,'VGGT\n预测几何 / 视觉特征','C9DBEE','86A7C8',15,bold=True)
    s.shape(4.75,1.35,2.15,.63,'预测相机 K、T','E8EFF7','A7BCD1',13)
    s.shape(4.75,2.44,2.15,.63,'深度 → 预测点云','E8EFF7','A7BCD1',13)
    s.shape(7.40,2.44,2.0,.63,'采样＋局部特征编码','FBE8D8','D6A373',12.5)
    s.tokens(9.72,2.62,False,'D39864','Point initialized reconstruction')
    s.text(9.48,2.16,2.0,.33,'重建token初始化',13,color=REC,bold=True)

    s.shape(2.05,4.10,2.25,1.10,'TokenGS / LocusGS\n图像编码器\nRGB＋预测Plücker射线','DBE6F3','93ACC6',13)
    s.shape(4.85,4.31,1.65,.69,'重建Memory\nF_rec','FBE5D0','D7AC80',13)
    s.shape(7.35,3.98,4.05,1.47,fill='FFFFFF',line='D5B89C',name='Reconstruction decoder')
    s.shape(7.58,4.28,1.47,.75,'Cross-Attn\nQ: H_rec\nK,V: F_rec','F5D8BA',None,12)
    s.shape(9.35,4.28,1.73,.75,'Self-Attn\n＋ FFN','F3E4D4',None,13)
    s.text(7.57,4.00,3.55,.27,'几何驱动的重建token更新',12,color=REC,bold=True)
    s.tokens(12.13,4.15,True,'CE8A4C','Reconstruction output')
    s.text(11.55,3.81,1.5,.3,'H_rec',14,color=REC,bold=True)
    s.shape(13.25,4.00,1.80,1.36,'Gaussian Head\n位置 / 尺度 / 旋转\n颜色 / Opacity','FBE8D7','D7AD83',12.5)
    s.shape(15.62,4.06,1.92,1.20,'3D Gaussians\nμ, s, R, α, c','FFF5EA','D7AD83',15,bold=True)

    s.shape(2.05,6.22,2.25,.74,'实例提议模型\nSAM / 其他模型（待定）','ECF3EB','84A18B',12.5,dash=True)
    s.shape(4.85,6.22,2.12,.74,'跨视图关联\n＋ Mask特征池化','EFF5EC','9DB394',12.5,dash=True)
    s.tokens(7.40,6.43,False,'83A778','Region initialized object')
    s.text(7.22,6.02,3.80,.30,'理解token初始化＋预留可学习槽位',12,color=SEM,bold=True)
    s.shape(4.85,7.43,2.12,1.05,'语义Memory F_sem\nVGGT视觉 / 多层特征\n＋语义适配器（候选）','E6DEEF','A497B8',12,dash=True)
    s.shape(7.35,7.35,4.05,1.41,fill='FFFFFF',line='A7BBA5',name='Object understanding decoder')
    s.text(7.55,7.37,3.62,.27,'实例驱动的理解token更新',12,color=SEM,bold=True)
    s.shape(7.56,7.82,1.03,.64,'CA视觉\nF_sem','DCEAD7',None,12)
    s.shape(8.86,7.82,1.03,.64,'CA重建\nH_rec','DCEAD7',None,12)
    s.shape(10.16,7.82,1.03,.64,'SA\n＋FFN','E7EEE2',None,12)
    s.tokens(12.13,7.53,True,'79A570','Object understanding output')
    s.text(11.55,7.12,1.5,.30,'H_obj',14,color=SEM,bold=True)
    s.shape(13.25,7.29,1.80,1.47,'实例读出\n逐Child语义特征\nMembership＋类别','E7F1E3','9EB49A',12.5)
    s.shape(15.62,7.43,1.92,1.20,'Panoptic渲染\n实例Mask\n语义标签','F0F6EC','9EB49A',14,bold=True)

    # Input branches and predicted cameras.
    s.line([(1.60,3.60),(1.80,3.60),(1.80,2.22),(2.05,2.22)],color=BLUE,name='RGB to VGGT')
    s.line([(1.60,4.00),(1.84,4.00),(1.84,4.65),(2.05,4.65)],name='RGB to reconstruction encoder')
    s.line([(1.60,5.00),(1.75,5.00),(1.75,6.59),(2.05,6.59)],color=SEM,name='RGB to proposal generator')
    s.line([(4.23,2.05),(4.48,2.05),(4.48,1.67),(4.75,1.67)],color=BLUE,name='Predicted cameras')
    s.line([(4.23,2.40),(4.48,2.40),(4.48,2.76),(4.75,2.76)],color=BLUE,name='Predicted point cloud')
    s.line([(5.83,1.98),(5.83,2.15),(3.15,2.15),(3.15,4.10)],color=BLUE,name='Predicted cameras to Plucker embedding')
    # Route camera line around VGGT box instead of implying cameras enter it.
    camera=s.tree[-1];s.tree.remove(camera)
    s.line([(5.83,1.98),(5.83,2.16),(4.40,2.16),(4.40,3.58),(3.15,3.58),(3.15,4.10)],color=BLUE,name='Pose free predicted cameras to Plucker rays')
    s.text(2.18,3.62,2.0,.30,'统一预测相机与坐标系',10.5,color=BLUE)
    s.line([(6.90,2.76),(7.40,2.76)],color=REC,name='Point cloud sampling initialization')
    s.line([(6.80,3.07),(6.80,6.22)],color=BLUE,dash=True,width=1.0,name='Predicted 3D support for cross view proposal association')
    s.line([(9.40,2.76),(9.70,2.76)],color=REC,name='Point content to reconstruction seed')
    s.line([(10.05,2.95),(10.05,3.18),(7.17,3.18),(7.17,4.43),(7.58,4.43)],color=REC,name='Reconstruction seed Q input')
    s.line([(4.30,4.65),(4.85,4.65)],color=REC,name='Encoder to reconstruction memory')
    s.line([(6.50,4.65),(7.58,4.65)],color=REC,name='F_rec keys and values')
    s.line([(9.05,4.65),(9.35,4.65)],color=REC,name='Reconstruction CA to SA')
    s.line([(11.40,4.65),(12.04,4.65)],color=REC,name='Updated reconstruction tokens')
    s.line([(12.50,4.65),(13.25,4.65)],color=REC,name='Reconstruction token to all Gaussian parameters')
    s.line([(15.05,4.65),(15.62,4.65)],color=REC,name='Gaussian reconstruction output')
    # Semantic initialization and memory.
    s.line([(4.30,6.59),(4.85,6.59)],color=SEM,dash=True,name='Automatic instance proposals')
    s.line([(5.90,7.43),(5.90,6.96)],color=SEM,dash=True,name='Mask pooling semantic content')
    s.line([(6.97,6.59),(7.38,6.59)],color=SEM,dash=True,name='Region pooled Object Query seeds')
    s.line([(7.90,6.72),(7.90,7.06),(7.17,7.06),(7.17,7.98),(7.56,7.98)],color=SEM,name='Object seed Q input')
    s.line([(4.23,2.57),(4.53,2.57),(4.53,7.95),(4.85,7.95)],color='8C78A5',dash=True,name='VGGT features candidate semantic memory')
    s.line([(6.97,8.13),(7.23,8.13),(7.23,8.27),(7.56,8.27)],color=SEM,name='Visual semantic memory keys values')
    s.line([(8.59,8.14),(8.86,8.14)],color=SEM,name='Visual evidence to geometric evidence')
    s.line([(9.89,8.14),(10.16,8.14)],color=SEM,name='Object cross attention to self attention')
    s.line([(11.40,8.14),(12.04,8.14)],color=SEM,name='Updated object tokens')
    s.line([(12.50,8.14),(13.25,8.14)],color=SEM,name='Object token to instance readout')
    s.line([(15.05,8.14),(15.62,8.14)],color=SEM,name='Panoptic output')
    # Asymmetric inter-stream evidence and optional next-stage feedback.
    s.line([(12.27,5.15),(12.27,6.96),(9.38,6.96),(9.38,7.82)],color=BLUE,width=1.7,name='Read reconstruction tokens as K V')
    s.text(9.08,6.63,3.00,.31,'Read：重建token提供K,V',11,color=BLUE,bold=True)
    s.line([(14.10,5.36),(14.10,7.29)],color=REC,name='Current Gaussian geometry to child descriptor')
    s.text(12.92,6.05,1.10,.58,'当前几何\n与可见性',10,color=REC)
    s.line([(6.97,8.35),(7.08,8.35),(7.08,9.20),(14.68,9.20),(14.68,8.76)],color=SEM,name='Fine visual features to child readout')
    s.text(9.80,8.91,3.90,.25,'细粒度视觉支撑 → 逐Child描述子',10,color=SEM)
    s.line([(15.05,7.57),(15.33,7.57),(15.33,5.60),(10.69,5.60),(10.69,5.45)],color=SEM,dash=True,width=1.8,name='Optional ownership gated Write to next reconstruction stage')
    s.text(7.67,5.70,4.25,.28,'可选Write：归属约束＋门控 → 下一阶段',10.5,color=SEM,bold=True)
    s.text(15.60,8.86,1.98,.32,'N_rec ≫ N_obj',11,color=GRAY)
    return s


def comparison():
    s=Slide();s.title('和 QuerySplat 相似在哪里，方法差异应画在哪里？',
        'CA / SA / FFN是共用算子；需要表达的是token职责、初始化、证据来源、交互与输出的变化。',2)
    s.shape(.55,1.52,16.90,.57,fill='EDF1F5',line=None,round_=False)
    cols=[(.72,3.1,'比较项'),(4.0,5.5,'原QuerySplat图'),(10.0,6.9,'我们的双流设计（待验证）')]
    for x,w,t in cols:s.text(x,1.61,w,.37,t,16,bold=True,align='l')
    rows=[
        ('两组token的职责','Geometry：位置/尺度/旋转\nAppearance：颜色与opacity','Reconstruction：完整Gaussian重建\nObject：实例归属与语义类别'),
        ('初始化','可学习Geometry Query\nGeometry token条件化Appearance Query','VGGT点云＋局部特征初始化重建\n实例候选池化＋可学习槽位初始化理解'),
        ('读取的场景证据','几何特征Memory\nRGB＋Plücker外观Memory','重建流读取RGB＋预测射线特征\n理解流读取视觉语义Memory与重建token'),
        ('两流关系','几何token条件化外观解码\n完成同一个重建任务','Read允许全局上下文\n可选Write按实例归属门控，作用于下一阶段'),
        ('输出','两条分支参数合成\n同一组3D Gaussians','重建参数由重建流统一产生\n理解流预测Child membership、类别与Mask'),
        ('需要验证的假设','原方法承担重建','初始化能否补足小物体覆盖？\n实例归属驱动交互是否改善Mask/PQ？'),
    ]
    for i,(a,b,c) in enumerate(rows):
        y=2.18+i*.91
        s.shape(.55,y,16.90,.83,fill='F7F9FA' if i%2==0 else 'FFFFFF',line=None,round_=False)
        s.text(.72,y+.10,3.00,.58,a,15,bold=True,align='l')
        s.text(4.00,y+.05,5.48,.73,b,14,color=GRAY,align='l')
        s.text(10.00,y+.05,6.85,.73,c,14,color=SEM,align='l')
    s.shape(.65,7.88,16.70,1.10,'相似是正常的，但点云 / SAM初始化本身还不足以证明方法贡献。\n主张应落到：实例级token的职责、细粒度归属读出、非对称交互；收益需要受控实验支持。',
        'F3F0E8','D4C8AE',15,bold=True)
    return s


def reference(source_root,old_w,old_h):
    s=Slide();s.title('QuerySplat 原始可编辑架构图（保留作对照）',
        '原文件内容未改写；本页仅将原始形状整体等比例放入新画布。',3)
    grp=el(s.tree,'p:grpSp');nv=el(grp,'p:nvGrpSpPr');el(nv,'p:cNvPr',id=60000,name='Original editable QuerySplat diagram')
    el(nv,'p:cNvGrpSpPr');el(nv,'p:nvPr')
    pr=el(grp,'p:grpSpPr');tr=el(pr,'a:xfrm')
    scale=min(17.10*EMU/old_w,7.65*EMU/old_h)
    el(tr,'a:off',x=units(.45),y=units(1.47));el(tr,'a:ext',cx=round(old_w*scale),cy=round(old_h*scale))
    el(tr,'a:chOff',x=0,y=0);el(tr,'a:chExt',cx=old_w,cy=old_h)
    for child in source_root.find('p:cSld/p:spTree',NS):
        if child.tag not in (tag('p:nvGrpSpPr'),tag('p:grpSpPr')):grp.append(copy.deepcopy(child))
    # Keep IDs unique after copying the source's native shapes.
    for i,node in enumerate(s.root.findall('.//p:cNvPr',NS),1):node.set('id',str(i))
    return s


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();assert a.source.resolve()!=a.output.resolve() and not a.output.exists()
    source_hash=hashlib.sha256(a.source.read_bytes()).hexdigest()
    with zipfile.ZipFile(a.source) as zin:
        files={n:zin.read(n) for n in zin.namelist()}
    p=E.fromstring(files['ppt/presentation.xml']);sz=p.find('p:sldSz',NS);oldw,oldh=int(sz.get('cx')),int(sz.get('cy'))
    source=E.fromstring(files['ppt/slides/slide1.xml']);slides=[architecture(),comparison(),reference(source,oldw,oldh)]
    sz.set('cx',units(W));sz.set('cy',units(H));sz.set('type','custom')
    ids=p.find('p:sldIdLst',NS);ids.clear()
    relns='http://schemas.openxmlformats.org/package/2006/relationships'
    rel=E.fromstring(files['ppt/_rels/presentation.xml.rels'])
    for child in list(rel):
        if child.get('Type','').endswith('/slide'):rel.remove(child)
    for i,slide in enumerate(slides,1):
        rid='rId'+str(100+i);el(ids,'p:sldId',{'id':255+i,tag('r:id'):rid})
        E.SubElement(rel,'{'+relns+'}Relationship',Id=rid,Type=NS['r']+'/slide',Target=f'slides/slide{i}.xml')
        files[f'ppt/slides/slide{i}.xml']=slide.xml()
        files[f'ppt/slides/_rels/slide{i}.xml.rels']=files['ppt/slides/_rels/slide1.xml.rels']
    files['ppt/presentation.xml']=E.tostring(p,encoding='utf-8',xml_declaration=True)
    E.register_namespace('',relns)
    files['ppt/_rels/presentation.xml.rels']=E.tostring(rel,encoding='utf-8',xml_declaration=True)
    ct=E.fromstring(files['[Content_Types].xml']);ctns='http://schemas.openxmlformats.org/package/2006/content-types'
    existing={x.get('PartName') for x in ct}
    for i in range(1,4):
        name=f'/ppt/slides/slide{i}.xml'
        if name not in existing:E.SubElement(ct,'{'+ctns+'}Override',PartName=name,ContentType='application/vnd.openxmlformats-officedocument.presentationml.slide+xml')
    E.register_namespace('',ctns)
    files['[Content_Types].xml']=E.tostring(ct,encoding='utf-8',xml_declaration=True)
    # Update slide count without changing the source package's editable theme.
    if 'docProps/app.xml' in files:
        app=E.fromstring(files['docProps/app.xml'])
        for item in app.iter():
            if item.tag.endswith('}Slides'):item.text='3'
            if item.tag.endswith('}PresentationFormat'):item.text='On-screen Show (16:9)'
        vt='http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes'
        E.register_namespace('vt',vt)
        counts=app.findall('.//{'+vt+'}i4')
        if len(counts)>=2:counts[1].text='3'
        vector=app.find('.//{http://schemas.openxmlformats.org/officeDocument/2006/extended-properties}TitlesOfParts/{'+vt+'}vector')
        if vector is not None:
            vector.clear();vector.set('size','4');vector.set('baseType','lpstr')
            for title in ['Office Theme','几何–语义双流总图','与QuerySplat的差异','QuerySplat原图参考']:
                E.SubElement(vector,'{'+vt+'}lpstr').text=title
        E.register_namespace('','http://schemas.openxmlformats.org/officeDocument/2006/extended-properties')
        files['docProps/app.xml']=E.tostring(app,encoding='utf-8',xml_declaration=True)
    if 'docProps/core.xml' in files:
        core=E.fromstring(files['docProps/core.xml'])
        prefixes={'dc':'http://purl.org/dc/elements/1.1/',
            'cp':'http://schemas.openxmlformats.org/package/2006/metadata/core-properties',
            'dcterms':'http://purl.org/dc/terms/','xsi':'http://www.w3.org/2001/XMLSchema-instance'}
        for prefix,uri in prefixes.items():E.register_namespace(prefix,uri)
        values={'dc:title':'几何–语义双流Gaussian架构（设计草案）',
            'dc:subject':'双流总图、方法差异与原QuerySplat图参考；可编辑原生形状',
            'dc:description':'Editable OOXML drawing; proposed architecture, not implemented or experimentally validated.',
            'cp:lastModifiedBy':'Codex','dcterms:modified':datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}
        for name,value in values.items():
            prefix,local=name.split(':');node=core.find('{'+prefixes[prefix]+'}'+local)
            if node is not None:node.text=value
        files['docProps/core.xml']=E.tostring(core,encoding='utf-8',xml_declaration=True)
    with zipfile.ZipFile(a.output,'w',zipfile.ZIP_DEFLATED) as zout:
        for name,data in files.items():zout.writestr(name,data)
    assert hashlib.sha256(a.source.read_bytes()).hexdigest()==source_hash
    audit=dict(source=str(a.source),source_sha256=source_hash,output=str(a.output),slides=3,
        output_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),
        builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        editable=True,native_shape_counts=[len(s.root.findall('.//p:sp',NS)) for s in slides],
        source_preserved=True,architecture_status='PROPOSAL_NOT_IMPLEMENTED',
        pending=['SAM or other proposal model','semantic memory source','ownership-gated write efficacy'])
    a.output.with_suffix('.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(audit,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
