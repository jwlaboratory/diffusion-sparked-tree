"""2-precompute charts: fast (serial per-pop CPU matmul) vs precompute (precomputed
table lookup), at a shared K, one budget. Grey = fast (the already-best transfer-less
builder, = the red bar from 1-transfer-less); red = precompute.

Usage:  python make_charts.py <summary.json> <K> <budget>
        (defaults: csweep/results/summary.json 256 64)
"""
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e6e3"
GREY, RED = "#9aa0a6", "#e34948"
PHASE_SEGS = [("draft_forward","draft fwd","#2a78d6"), ("cb_prep","cand .prep","#e6a13c"),
              ("cb_expand","cand .expand","#e34948"), ("verify","verify","#7b5cd6"),
              ("other","other","#9a9a97")]
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": GRID, "xtick.color": INK2,
    "ytick.color": INK2, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

def load(path, K, b):
    s = json.loads(Path(path).read_text())
    return s, str(b), f"fast.c{K}", f"precompute.c{K}", K

def accept(s, b, arm):
    a = r = 0.0
    for ds in s["results"]["clean"][b]:
        e = s["results"]["clean"][b][ds][arm]; a += e["mean_accept"]*e["rounds"]; r += e["rounds"]
    return a/r
def tps(s, b, arm): return s["timing"][b][arm]["tps_clean"]
def ms_round(s, b, arm):
    inst = s["results"]["instrumented"][b]
    R=0.0; P={}; U={}
    for ds in inst:
        e = inst[ds][arm]; R += e["rounds"]
        for k,v in (e.get("phases") or {}).items():
            if v: P[k]=P.get(k,0)+v["sec"]
        for k,v in (e.get("subphases") or {}).items():
            if v: U[k]=U.get(k,0)+v["sec"]
    f=1000.0/R; tot=sum(P.values())
    return {"draft_forward":P.get("draft_forward",0)*f, "cb_prep":U.get("candidate_build.prep",0)*f,
            "cb_expand":U.get("candidate_build.expand",0)*f, "verify":P.get("verify",0)*f,
            "other":max(tot-P.get("draft_forward",0)-P.get("verify",0)-U.get("candidate_build.prep",0)-U.get("candidate_build.expand",0),0)*f,
            "_total":tot*f}

def fig_speedup(s, b, fast, pre, K, out):
    fig,(axL,axR)=plt.subplots(1,2,figsize=(9,4.6)); w=0.38
    tf,tp=tps(s,b,fast),tps(s,b,pre); af,ap=accept(s,b,fast),accept(s,b,pre)
    for ax,vf,vp,fmt,ttl,ymax in [(axL,tf,tp,"{:.0f}","Speed",max(tf,tp)*1.28),
                                   (axR,af,ap,"{:.2f}","Acceptance",max(af,ap)*1.25)]:
        ax.grid(axis="y",color=GRID,lw=1); ax.set_axisbelow(True)
        ax.bar(-w/2,vf,w,color=GREY,edgecolor=SURFACE,linewidth=1.5,zorder=3)
        ax.bar(w/2,vp,w,color=RED,edgecolor=SURFACE,linewidth=1.5,zorder=3)
        ax.text(-w/2,vf,fmt.format(vf),ha="center",va="bottom",fontsize=10,color=INK2)
        ax.text(w/2,vp,fmt.format(vp),ha="center",va="bottom",fontsize=10,color=INK2)
        ax.set_xticks([0]); ax.set_xticklabels([f"budget {b}"]); ax.set_ylim(0,ymax)
        ax.set_title(ttl,fontsize=13,fontweight="bold",loc="left")
    axL.set_ylabel("net decode throughput (tokens/s, clean)")
    axR.set_ylabel("mean acceptance (tokens/round)")
    axL.annotate(f"{tp/tf:.2f}× TPS",(w/2,tp),(w/2,tp+tf*0.13),ha="center",fontsize=11,fontweight="bold",color=RED)
    axR.annotate(f"{100*(ap-af)/af:+.1f}% accept",(0,max(af,ap)),(0,max(af,ap)+af*0.12),ha="center",fontsize=11,fontweight="bold",color=INK2)
    fig.legend(handles=[Patch(facecolor=GREY,label="fast — serial per-pop CPU matmul (transfer-less)"),
                        Patch(facecolor=RED,label="precompute — one upfront matmul + table lookup")],
               loc="lower center",ncol=1,frameon=False,fontsize=9.5,bbox_to_anchor=(0.5,-0.04))
    fig.suptitle(f"Precompute vs the transfer-less best (budget {b}, top-K={K})",fontsize=13.5,fontweight="bold",x=0.02,ha="left")
    fig.text(0.02,0.90,"SparklingTree_b16, h100, bench x8. Same builder inputs; only where the markov arithmetic runs differs.",fontsize=9,color=INK2)
    fig.tight_layout(rect=(0,0.06,1,0.92)); fig.savefig(out,dpi=150,bbox_inches="tight"); print("wrote",out)

def fig_phase(s, b, fast, pre, K, out):
    fig,ax=plt.subplots(figsize=(11,3.4)); ax.grid(axis="x",color=GRID,lw=1); ax.set_axisbelow(True)
    rows=[(fast,"fast",0),(pre,"precompute",1)]  # y=0 bottom=fast, y=1 top=precompute (matches yticklabels)
    xmax=max(ms_round(s,b,fast)["_total"],ms_round(s,b,pre)["_total"])*1.16
    for y,(arm,lab,_) in enumerate(rows):
        seg=ms_round(s,b,arm); left=0.0
        for key,name,col in PHASE_SEGS:
            v=seg[key]; ax.barh(y,v,left=left,height=0.5,color=col,edgecolor=SURFACE,linewidth=1.5,zorder=3)
            if v>seg["_total"]*0.05: ax.text(left+v/2,y,f"{v:.0f}",ha="center",va="center",fontsize=8.5,color="white",fontweight="bold",zorder=4)
            left+=v
        ax.text(left+xmax*0.005,y,f"{seg['_total']:.0f} ms/round",ha="left",va="center",fontsize=9.5,color=INK,fontweight="bold")
    ax.set_yticks([0,1]); ax.set_yticklabels(["fast\n(per-pop matmul)","precompute\n(table lookup)"]); ax.set_ylim(-0.6,1.6); ax.set_xlim(0,xmax)
    ef,ep=ms_round(s,b,fast)["cb_expand"],ms_round(s,b,pre)["cb_expand"]
    tf2,tp2=ms_round(s,b,fast)["_total"],ms_round(s,b,pre)["_total"]
    ax.set_title(f"budget {b}, top-K={K} — precompute drives .expand {ef:.1f}→{ep:.2f} ms/round (total {tf2:.0f}→{tp2:.0f}, ~{100*(tf2-tp2)/tf2:.0f}%), a small win at b64",fontsize=11,fontweight="bold",loc="left")
    ax.set_xlabel("decode time per round (ms), instrumented pass")
    fig.legend(handles=[Patch(facecolor=c,label=n) for _,n,c in PHASE_SEGS],loc="lower center",ncol=5,frameon=False,fontsize=9.5,bbox_to_anchor=(0.5,-0.05))
    fig.tight_layout(rect=(0,0.08,1,1)); fig.savefig(out,dpi=150,bbox_inches="tight"); print("wrote",out)

if __name__=="__main__":
    path=sys.argv[1] if len(sys.argv)>1 else str(HERE/"csweep"/"results"/"summary.json")
    K=int(sys.argv[2]) if len(sys.argv)>2 else 256
    b=sys.argv[3] if len(sys.argv)>3 else "64"
    s,bb,fast,pre,K=load(path,K,b)
    outdir=HERE/"results"; outdir.mkdir(exist_ok=True)
    fig_speedup(s,bb,fast,pre,K,outdir/"speedup_acceptance.png")
    fig_phase(s,bb,fast,pre,K,outdir/"phase_collapse.png")
