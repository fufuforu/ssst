"""Build Chinese presentation figures from saved C predictions; no model forward."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

FONT = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
font_manager.fontManager.addfont(str(FONT))
plt.rcParams.update({'font.family': font_manager.FontProperties(fname=str(FONT)).get_name(), 'axes.unicode_minus': False,
    'pdf.fonttype': 42, 'font.size': 12, 'axes.titleweight': 'bold'})
INK, MUTED, TEAL, ORANGE, RED, BLUE = '#18344b', '#536779', '#16877f', '#d88b35', '#d7524c', '#2e84ca'
CLASS_NAMES = ['柜子','床','椅子','沙发','桌子','门','窗','书架','画','台面','书桌','窗帘',
    '冰箱','浴帘','马桶','洗手池','浴缸','其他家具','no-object']


def read(p):
    return json.loads(p.read_text())


def base(title, subtitle, page):
    fig = plt.figure(figsize=(16,9), facecolor='white')
    fig.text(.045,.935,title,fontsize=25,color=INK,weight='bold')
    fig.text(.045,.884,subtitle,fontsize=12,color=MUTED)
    fig.text(.045,.027,'固定权重 C｜128个留出场景｜492个GT实例｜仅重绘已有产物，无训练或新推理',fontsize=9,color=MUTED)
    fig.text(.95,.027,f'{page} / 3',fontsize=10,color=MUTED,ha='right')
    return fig


def bbox(mask):
    yy,xx=np.where(mask)
    x0,x1,y0,y1=xx.min(),xx.max()+1,yy.min(),yy.max()+1
    side=min(256,max(x1-x0,y1-y0,64)+28)
    cx,cy=(x0+x1)/2,(y0+y1)/2
    x0=int(np.clip(cx-side/2,0,256-side));y0=int(np.clip(cy-side/2,0,256-side))
    return (x0,x0+side,y0,y0+side)


def image(ax,rgb,gt,pred=None,crop=None):
    ax.imshow(rgb)
    if pred is None:
        ax.contour(gt,levels=[.5],colors=['#ffe169'],linewidths=1.5)
    else:
        rgba=np.zeros((*gt.shape,4))
        rgba[pred & gt]=[.05,.85,.38,.63]
        rgba[pred & ~gt]=[.95,.15,.15,.65]
        rgba[~pred & gt]=[.08,.45,1,.65]
        ax.imshow(rgba)
    if crop:
        x0,x1,y0,y1=crop;ax.set_xlim(x0,x1);ax.set_ylim(y1,y0)
    ax.axis('off')


def load_case(root,pairs,name,iid,mode):
    path=next((root/'C').glob('rank*/'+name+'.json'));r=read(path)
    g=r['gt_ids'].index(iid)
    raw=np.array(r['mask_IoU']['raw_all'])[:,g]
    q=int(raw.argmax()) if mode=='best_raw' else r['matched_query'][r['matched_gt'].index(g)]
    qfinal=int(np.array(r['mask_IoU']['final'])[:,g].argmax()) if mode=='best_raw' else q
    masks=[]
    for frame in r['frame_ids'][:2]:
        a=np.array(Image.open(pairs[name]/'context_seg_gt'/f'{name.split("_context")[0]}_gt{frame}.png'),dtype=np.int64)
        packed=a[...,0]+256*a[...,1]+65536*a[...,2]
        masks.append(packed%1000==iid)
    view=int(np.argmax([m.sum() for m in masks]));gt=masks[view]
    rgb=np.array(Image.open(pairs[name]/'rgb_gt'/f'{name.split("_context")[0]}_{r["frame_ids"][view]}.png'))/255
    with np.load(path.with_suffix('.npz')) as z:
        rawmask=np.unpackbits(z['raw_context'][view,q],axis=-1).astype(bool)
        final=z['final_context'][view]==qfinal+1
    pc=np.array(r['query_class_probabilities'][q]);pred=int(pc.argmax())
    data=dict(name=name,GT_id=iid,GT_class=CLASS_NAMES[r['gt_classes'][g]-2],query=q,final_query=qfinal,
        predicted_class=CLASS_NAMES[pred],raw_IoU=float(raw[q]),final_IoU=float(r['mask_IoU']['final'][qfinal][g]),
        final_best_IoU=float(np.array(r['mask_IoU']['final'])[:,g].max()),GT_pixels=r['gt_pixels'][g],
        frame_id=r['frame_ids'][view],query_selection=mode,source_json=str(path))
    return rgb,gt,rawmask,final,pc,data


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    results=read(a.root/'analysis/results.json')
    audit=read(a.root/'analysis/completion_audit_and_sensitivity.json')
    assert audit['status']=='COMPLETE_256_WINDOWS_AUDITED'
    ev=Path('/space/mawb/ssst/group_plus/vggt_classification_readback_v1/attempts/attempt01/object_support/evaluation')
    pairs={r['name']:p.parent/'all'/r['name'] for p in ev.glob('rank*/export_complete.json') for r in read(p)['records']}
    manifest=[]
    with PdfPages(a.output/'Mask瓶颈诊断_汇报图.pdf') as pdf:
        fig=base('当前实例问题集中在两个环节','问题一：没有形成足够好的候选Mask　　问题二：已有好Mask，却未被正确分类和保留',1)
        fig.text(.07,.795,'① Raw Mask覆盖不足，小物体最明显',fontsize=18,color=INK,weight='bold')
        ax=fig.add_axes([.09,.36,.36,.32])
        small=audit['stage_groups']['C']['groups']['small'];all_=audit['stage_groups']['C']['groups']['all']
        good_small=small['raw_all']['best50'];good_other=all_['raw_all']['best50']-good_small
        sizes=[small['n'],all_['n']-small['n']];good=[good_small,good_other]
        for y,(n,k) in enumerate(zip(sizes,good)):
            ax.barh(y,k/n,color=TEAL,height=.42)
            ax.barh(y,(n-k)/n,left=k/n,color='#f0d7b4',height=.42)
            ax.text(k/n/2,y,f'{k}/{n}\n{k/n:.1%}',ha='center',va='center',color='white',fontsize=14,weight='bold')
            ax.text(k/n+(n-k)/n/2,y,f'未达标 {n-k}',ha='center',va='center',color=INK,fontsize=11)
        ax.set_yticks([0,1],['小物体','其余实例']);ax.invert_yaxis();ax.set_xticks([0,.5,1],['0%','50%','100%'])
        ax.set_xlim(0,1);ax.set_xlabel('至少一个Raw候选达到 IoU ≥ 0.5 的比例')
        for spine in ax.spines.values():spine.set_visible(False)
        fig.text(.08,.27,'134 / 492 个实例没有任何合格Raw候选',fontsize=17,color=ORANGE,weight='bold')
        fig.text(.08,.22,'其中85个是小物体；失败发生在最终分类之前。',fontsize=12,color=MUTED)
        fig.text(.08,.165,'小物体：两输入帧GT总面积占比 < 1%。',fontsize=10,color=MUTED)
        fig.text(.55,.795,'② 好候选在分类 / 筛选阶段损失',fontsize=18,color=INK,weight='bold')
        ax=fig.add_axes([.56,.36,.36,.32]);names=['Raw候选','资格筛选后','最终Mask'];values=[358,294,288]
        ax.bar(range(3),values,color=[TEAL,ORANGE,BLUE],width=.58)
        for i,v in enumerate(values):ax.text(i,v+10,str(v),ha='center',fontsize=20,color=INK,weight='bold')
        ax.set_xticks(range(3),names);ax.set_ylim(0,450);ax.set_ylabel('具有合格几何Mask的GT实例数')
        ax.spines[['top','right']].set_visible(False)
        fig.text(.56,.27,'Raw后净减少70个，64个发生在资格筛选',fontsize=16,color=ORANGE,weight='bold')
        fig.text(.56,.22,'原生匹配的好Mask：49个no-object，24个类别错。',fontsize=12,color=MUTED)
        fig.text(.56,.165,'按GT取最好Query，已考虑其他候选接替；不等同于AP/PQ。',fontsize=10,color=MUTED)
        fig.savefig(a.output/'01_两个主要问题_量化概览.png',dpi=180);pdf.savefig(fig);plt.close(fig)

        fig=base('问题一：最终分类之前，就没有形成好Mask','展示远处小物体与遮挡实例；即使事后在全部100个Query中选最好Raw候选，IoU仍低于0.5',2)
        raw_cases=[('scene0500_01_context699_710',36,'远处小物体'),('scene0342_00_context315_338',12,'遮挡椅子')]
        for row,(name,iid,label) in enumerate(raw_cases):
            rgb,gt,raw,final,pc,data=load_case(a.root,pairs,name,iid,'best_raw');manifest.append(data)
            assert data['raw_IoU']<.5
            y=.48-row*.355;crop=bbox(gt)
            for col in range(4):
                ax=fig.add_axes([.075+col*.19,y,.175,.275])
                if col==0:
                    image(ax,rgb,gt);x0,x1,y0,y1=crop
                    ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='#5fe6eb',linewidth=1.4))
                    title='原图与GT轮廓'
                elif col==1:
                    image(ax,rgb,gt,crop=crop);title='局部GT（黄色轮廓）'
                elif col==2:
                    image(ax,rgb,gt,raw,crop);title=f'最好Raw候选：{data["raw_IoU"]:.3f}'
                else:
                    image(ax,rgb,gt,final,crop);title=f'最好最终Mask：{data["final_best_IoU"]:.3f}'
                ax.set_title(title,fontsize=12,color=INK,pad=9)
            fig.text(.86,y+.19,label,fontsize=15,color=INK,weight='bold')
            fig.text(.86,y+.12,f'GT：{data["GT_class"]}\nGT面积占比\n{data["GT_pixels"]/(2*256*256):.2%}',fontsize=12,color=MUTED)
            fig.text(.075,y-.035,f'{name}  /  GT {iid}',fontsize=9,color=MUTED)
        fig.text(.075,.065,'绿色：预测正确部分　红色：误预测部分　蓝色：漏掉的GT　｜IoU为两输入帧联合值；图片为一帧局部展示。',fontsize=10,color=MUTED)
        fig.savefig(a.output/'02_RawMask形成不足_真实案例.png',dpi=180);pdf.savefig(fig);plt.close(fig)

        fig=base('问题二：好Mask已存在，类别与保留决策仍会失败','上：no-object导致候选被删除　　下：几何Mask准确，但类别错误；两个问题都无法只靠Mask IoU解释',3)
        head_cases=[('scene0050_00_context1441_1460',2,'好Mask被删'),('scene0144_01_context1068_1086',9,'好Mask类别错')]
        for row,(name,iid,label) in enumerate(head_cases):
            rgb,gt,raw,final,pc,data=load_case(a.root,pairs,name,iid,'native_matched');manifest.append(data)
            assert data['raw_IoU']>.9
            y=.48-row*.355
            for col in (0,1,3):
                ax=fig.add_axes([.075+col*.19,y,.175,.275])
                image(ax,rgb,gt,None if col==0 else (raw if col==1 else final))
                ax.set_title('原图与GT轮廓' if col==0 else (f'Raw Mask：{data["raw_IoU"]:.3f}' if col==1 else f'该候选最终Mask：{data["final_IoU"]:.3f}'),fontsize=12,color=INK,pad=9)
            ax=fig.add_axes([.475,y+.015,.145,.23]);gt_cls=CLASS_NAMES.index(data['GT_class']);pred=int(pc.argmax())
            ids=list(dict.fromkeys([pred,gt_cls,18]));vals=[pc[i] for i in ids]
            ax.barh(range(len(ids)),vals,color=[RED if i==pred and i!=gt_cls else (TEAL if i==gt_cls else '#a9b3be') for i in ids],height=.42)
            ax.set_yticks(range(len(ids)),[CLASS_NAMES[i] for i in ids],fontsize=10);ax.invert_yaxis();ax.set_xlim(0,1)
            ax.set_xticks([0,.5,1]);ax.tick_params(axis='x',labelsize=9);ax.set_title('原分类头概率',fontsize=12,color=INK,pad=9)
            ax.spines[['top','right','left']].set_visible(False)
            for i,v in enumerate(vals):ax.text(min(v+.025,.87),i,f'{v:.2f}',va='center',fontsize=10)
            fig.text(.86,y+.19,label,fontsize=15,color=RED,weight='bold')
            fig.text(.86,y+.115,f'GT：{data["GT_class"]}\n预测：{data["predicted_class"]}',fontsize=12,color=MUTED)
            if row==0:fig.text(.86,y+.035,f'其他候选也未补救\n最终最好IoU\n{data["final_best_IoU"]:.3f}',fontsize=10,color=MUTED)
            fig.text(.075,y-.035,f'{name}  /  GT {iid}  /  Query {data["query"]}',fontsize=9,color=MUTED)
        fig.text(.075,.065,'颜色仅表示几何重叠：绿色不代表类别正确。IoU为两输入帧联合值；案例按GT事后选取，仅用于解释现象。',fontsize=10,color=MUTED)
        fig.savefig(a.output/'03_好Mask筛选与分类失败_真实案例.png',dpi=180);pdf.savefig(fig);plt.close(fig)

    source_files=[a.root/'analysis/results.json',a.root/'analysis/completion_audit_and_sensitivity.json',Path(__file__)]
    (a.output/'manifest.json').write_text(json.dumps(dict(status='PRESENTATION_EXPORTED_FROM_EXISTING_PREDICTIONS',
        model='C',scenes=128,GT_instances=492,new_training_updates=0,new_model_forwards=0,cases=manifest,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        scope='Two observed mask/selection bottlenecks. Extra 300-step continuation is a control, not evidence against the standard adapt2+joint4 schedule.'),indent=2,ensure_ascii=False)+'\n')
    print(a.output)


if __name__=='__main__':main()
