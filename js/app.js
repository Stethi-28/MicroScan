/* Microplastic Bench: particle detection and classification (built-in rules or a YOLOv8/YOLO11 ONNX model). */
(()=>{
"use strict";
const $=s=>document.querySelector(s);
let CLS={}; // filled from the YOLO11 model's class names
const EXCL="#9aa3a9";
const DEF_RULES={fiber:3,foam:0.06,bead:0.78,film:0.2};
const SIZE_BINS=[[0,100,"<100 µm"],[100,300,"100–300 µm"],[300,500,"300–500 µm"],[500,1000,"0.5–1 mm"],[1000,5000,"1–5 mm"],[5000,Infinity,">5 mm meso"]];

const S={base:null,W:0,H:0,rgba:null,d:null,lab:null,labIdx:null,scale:1,origW:0,origH:0,name:"",example:false,
  parts:[],sel:-1,T:{hi:0,lo:0},view:{outline:true,ids:true,mask:false},sort:{k:"id",dir:1},filter:null,
  set:{mode:"bright",bg:true,auto:true,thr:40,sens:1,hyst:0.4,minUm:20,edge:true},
  rules:{...DEF_RULES},umPerPx:3.5,cal:null,outline:null,ms:0};

/* ---------- persistence of method settings (per-viewer convenience) ---------- */
function loadPrefs(){try{const p=JSON.parse(localStorage.getItem("mpb-prefs")||"null");if(p){Object.assign(S.set,p.set||{});Object.assign(S.rules,p.rules||{});if(p.umPerPx>0)S.umPerPx=p.umPerPx;}}catch(e){}}
function savePrefs(){try{localStorage.setItem("mpb-prefs",JSON.stringify({set:S.set,rules:S.rules,umPerPx:S.umPerPx}));}catch(e){}}

/* ---------- image processing ---------- */
function convexHull(keys,H1){
  const P=[];let last=-1;for(let i=0;i<keys.length;i++){const k=keys[i];if(k===last)continue;last=k;P.push([Math.floor(k/H1),k%H1]);}
  if(P.length<3)return P;
  const cr=(o,a,b)=>(a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0]);
  const lo=[];for(const p of P){while(lo.length>=2&&cr(lo[lo.length-2],lo[lo.length-1],p)<=0)lo.pop();lo.push(p);}
  const up=[];for(let i=P.length-1;i>=0;i--){const p=P[i];while(up.length>=2&&cr(up[up.length-2],up[up.length-1],p)<=0)up.pop();up.push(p);}
  up.pop();lo.pop();return lo.concat(up);
}
function hullMetrics(h){
  let area=0,fmax=0,fmin=Infinity;const n=h.length;
  if(n<3){return{area:1,fmax:1,fmin:1};}
  for(let i=0;i<n;i++){const a=h[i],b=h[(i+1)%n];area+=a[0]*b[1]-b[0]*a[1];}
  area=Math.abs(area)/2;
  for(let i=0;i<n;i++)for(let j=i+1;j<n;j++){const dx=h[i][0]-h[j][0],dy=h[i][1]-h[j][1],d=dx*dx+dy*dy;if(d>fmax)fmax=d;}
  fmax=Math.sqrt(fmax);
  for(let i=0;i<n;i++){const a=h[i],b=h[(i+1)%n],ex=b[0]-a[0],ey=b[1]-a[1],L=Math.hypot(ex,ey);if(!L)continue;let m=0;
    for(let j=0;j<n;j++){const d=Math.abs(ex*(h[j][1]-a[1])-ey*(h[j][0]-a[0]))/L;if(d>m)m=d;}if(m<fmin)fmin=m;}
  return{area,fmax,fmin:isFinite(fmin)?fmin:1};
}
function colorName(r,g,b,mode){
  if(mode==="fluor")return"fluorescent";
  const mx=Math.max(r,g,b),mn=Math.min(r,g,b),v=mx/255,s=mx?(mx-mn)/mx:0;
  if(s<0.2||mx-mn<22){if(v<0.38)return"black";if(v>0.74)return"white / clear";return"grey";}
  if(v<0.22)return"black";
  let h;const c=mx-mn;
  if(mx===r)h=60*(((g-b)/c)%6);else if(mx===g)h=60*((b-r)/c+2);else h=60*((r-g)/c+4);if(h<0)h+=360;
  if(h<15||h>=345)return"red";if(h<45)return v<0.6?"brown":"orange";if(h<70)return"yellow";if(h<165)return"green";if(h<255)return"blue";if(h<290)return"purple";return"pink";
}
const COLOR_SW={"black":"#1b1b1b","white / clear":"#f3f3f0","grey":"#8c8c8c","red":"#d23a2f","brown":"#7a4a22","orange":"#ea7a1c","yellow":"#e9c42c","green":"#2f9e44","blue":"#2b6fd1","purple":"#7a4bc2","pink":"#e05a9c","fluorescent":"#ff8a3d"};

function analyze(){
  if(!S.base)return Promise.resolve();
  return M.ready.then(()=>M.session?analyzeYolo():undefined);
}

function features(c,lab,q,d,umPx){
  const {W,H,rgba}=S,L=c.L;
  let n=0,sx=0,sy=0,sr=0,sg=0,sb=0,sd=0,sdd=0,edges=0,minx=W,miny=H,maxx=0,maxy=0;
  const corners=[];const H1=H+1;
  for(let j=c.start;j<c.end;j++){
    const p=q[j],x=p%W,y=(p-x)/W;n++;sx+=x;sy+=y;
    const k=p*4;sr+=rgba[k];sg+=rgba[k+1];sb+=rgba[k+2];const dv=d[p];sd+=dv;sdd+=dv*dv;
    if(x<minx)minx=x;if(x>maxx)maxx=x;if(y<miny)miny=y;if(y>maxy)maxy=y;
    let e=0;if(x===0||lab[p-1]!==L)e++;if(x===W-1||lab[p+1]!==L)e++;if(y===0||lab[p-W]!==L)e++;if(y===H-1||lab[p+W]!==L)e++;
    if(e){edges+=e;corners.push(x*H1+y,(x+1)*H1+y,x*H1+y+1,(x+1)*H1+y+1);}
  }
  const keys=Int32Array.from(corners).sort();
  const hull=convexHull(keys,H1);const hm=hullMetrics(hull);
  // holes inside the bounding box
  const bw=maxx-minx+1,bh=maxy-miny+1,vis=new Uint8Array(bw*bh);
  for(let j=c.start;j<c.end;j++){const p=q[j],x=p%W,y=(p-x)/W;vis[(y-miny)*bw+(x-minx)]=1;}
  const st=[];
  const push=(i)=>{if(vis[i]===0){vis[i]=2;st.push(i);}};
  for(let x=0;x<bw;x++){push(x);push((bh-1)*bw+x);}
  for(let y=0;y<bh;y++){push(y*bw);push(y*bw+bw-1);}
  while(st.length){const i=st.pop(),x=i%bw,y=(i-x)/bw;if(x>0)push(i-1);if(x<bw-1)push(i+1);if(y>0)push(i-bw);if(y<bh-1)push(i+bw);}
  let holes=0;for(let i=0;i<vis.length;i++)if(vis[i]===0)holes++;
  const filled=n+holes;
  // outer perimeter only (pixel edges facing the exterior, not holes)
  let outer=0;
  for(let j=c.start;j<c.end;j++){const p=q[j],x=p%W,y=(p-x)/W,lx=x-minx,ly=y-miny,i=ly*bw+lx;
    if(lx===0||vis[i-1]===2)outer++;if(lx===bw-1||vis[i+1]===2)outer++;if(ly===0||vis[i-bw]===2)outer++;if(ly===bh-1||vis[i+bw]===2)outer++;}
  const P=outer*Math.PI/4;
  const fmax=Math.max(1,hm.fmax),fmin=Math.max(1,hm.fmin);
  const curv=Math.max(fmax,P/2);
  const meanD=sd/n,stdD=Math.sqrt(Math.max(0,sdd/n-meanD*meanD));
  const r=sr/n,g=sg/n,b=sb/n;
  return{L,n,filled,holes,cx:sx/n,cy:sy/n,minx,miny,maxx,maxy,
    edge:minx===0||miny===0||maxx===W-1||maxy===H-1,
    perim:P,fmax,fmin,ar:fmax/fmin,curv,fiberIdx:curv*curv/filled,
    circ:Math.min(1,4*Math.PI*filled/(P*P)),solidity:Math.min(1,filled/Math.max(hm.area,1)),
    roundness:Math.min(1,4*filled/(Math.PI*fmax*fmax)),holeFrac:holes/filled,
    meanD,contrast:meanD/255,texture:stdD/(meanD+1),rgb:[r,g,b].map(Math.round),
    color:colorName(r,g,b,S.set.mode),override:null,manualExcl:null};
}

const clamp=(v,a,b)=>v<a?a:v>b?b:v;
function classifyAll(){
  const umPx=S.umPerPx*S.scale;
  for(const p of S.parts){
    const c=p.modelCls,conf=p.score;
    p.auto=c;p.conf=conf;p.cls=p.override||c;
    p.excl=null;
  }
}
function um(p){
  const u=S.umPerPx*S.scale;const fiber=p.cls==="fiber"||!!CLS[p.cls]?.fiber;
  const len=(fiber?p.curv:p.fmax)*u;
  return{len,wid:fiber?(p.n/p.curv)*u:p.fmin*u,fmax:p.fmax*u,fmin:p.fmin*u,ecd:2*Math.sqrt(p.filled/Math.PI)*u,area:p.filled*u*u};
}
const confLabel=c=>c>=0.8?["High","h"]:c>=0.65?["Med","m"]:["Low","l"];

/* ---------- rendering ---------- */
const view=$("#view"),vctx=view.getContext("2d");
function hexRgb(h){const v=parseInt(h.slice(1),16);return[v>>16&255,v>>8&255,v&255];}
function buildOutline(){
  const {W,H,lab,labIdx,parts}=S;if(!lab)return;
  const img=new ImageData(W,H),o=img.data;
  const cols=parts.map(p=>hexRgb(p.excl?EXCL:CLS[p.cls].color));
  const thick=W>900;const mask=S.view.mask,outl=S.view.outline;
  const paint=(i,c,a)=>{const k=i*4;o[k]=c[0];o[k+1]=c[1];o[k+2]=c[2];o[k+3]=a;};
  for(let y=0;y<H;y++)for(let x=0;x<W;x++){
    const i=y*W+x,L=lab[i];if(!L)continue;const k=labIdx[L];if(k<0)continue;const c=cols[k];
    const b=(x===0||lab[i-1]!==L)||(x===W-1||lab[i+1]!==L)||(y===0||lab[i-W]!==L)||(y===H-1||lab[i+W]!==L);
    if(b&&outl){paint(i,c,255);if(thick){if(x>0&&lab[i-1]!==L)paint(i-1,c,255);if(x<W-1&&lab[i+1]!==L)paint(i+1,c,255);if(y>0&&lab[i-W]!==L)paint(i-W,c,255);if(y<H-1&&lab[i+W]!==L)paint(i+W,c,255);}}
    else if(mask&&o[i*4+3]===0)paint(i,c,150);
  }
  if(!S.outline){S.outline=document.createElement("canvas");}
  S.outline.width=W;S.outline.height=H;S.outline.getContext("2d").putImageData(img,0,0);
}
function drawAnnotated(ctx,sc,opts){
  const {W,H}=S;
  ctx.save();ctx.scale(sc,sc);
  if(S.view.mask&&!opts.export){ctx.fillStyle="#0b1114";ctx.fillRect(0,0,W,H);ctx.globalAlpha=0.35;ctx.drawImage(S.base,0,0);ctx.globalAlpha=1;}
  else ctx.drawImage(S.base,0,0);
  if(S.outline&&(S.view.outline||S.view.mask))ctx.drawImage(S.outline,0,0);
  if(S.view.ids||opts.export){
    const fs=Math.max(11,Math.round(W/95));ctx.font=`500 ${fs}px "IBM Plex Mono",monospace`;ctx.textBaseline="bottom";ctx.lineJoin="round";
    for(const p of S.parts){const t=String(p.id);const x=Math.min(p.maxx+2,W-fs*1.8),y=Math.max(p.miny,fs);
      ctx.lineWidth=Math.max(2.5,fs/4);ctx.strokeStyle="rgba(0,0,0,.8)";ctx.strokeText(t,x,y);ctx.fillStyle=p.excl?EXCL:CLS[p.cls].color;ctx.fillText(t,x,y);}
  }
  if(!opts.export&&S.sel>=0){const p=S.parts[S.sel],pad=Math.max(6,W/150);
    ctx.lineWidth=Math.max(2,W/400);ctx.strokeStyle="#ffffff";ctx.setLineDash([Math.max(4,W/150),Math.max(3,W/250)]);
    ctx.strokeRect(p.minx-pad,p.miny-pad,p.maxx-p.minx+1+2*pad,p.maxy-p.miny+1+2*pad);ctx.setLineDash([]);}
  if(!opts.export&&S.cal){ctx.strokeStyle="#ffb020";ctx.fillStyle="#ffb020";ctx.lineWidth=Math.max(2,W/400);const r=Math.max(4,W/200);
    for(const pt of S.cal.pts){ctx.beginPath();ctx.arc(pt[0],pt[1],r,0,Math.PI*2);ctx.fill();}
    if(S.cal.pts.length===2){ctx.beginPath();ctx.moveTo(...S.cal.pts[0]);ctx.lineTo(...S.cal.pts[1]);ctx.stroke();}}
  // scale bar on exports
  if(opts.export){const u=S.umPerPx*S.scale;const nice=[10,20,50,100,200,500,1000,2000,5000];let L=nice[0];for(const v of nice)if(v/u<=W*0.2)L=v;
    const px=L/u,x0=W*0.03,y0=H*0.95,fs=Math.max(12,Math.round(W/70));ctx.fillStyle="rgba(0,0,0,.6)";ctx.fillRect(x0-8,y0-fs-14,px+16,fs+22);
    ctx.fillStyle="#fff";ctx.fillRect(x0,y0,px,Math.max(3,H/250));ctx.font=`500 ${fs}px "IBM Plex Mono",monospace`;ctx.textBaseline="bottom";ctx.fillText(L>=1000?(L/1000)+" mm":L+" µm",x0,y0-4);}
  ctx.restore();
}
function drawView(){
  if(!S.base){vctx.clearRect(0,0,view.width,view.height);return;}
  if(view.width!==S.W||view.height!==S.H){view.width=S.W;view.height=S.H;}
  vctx.clearRect(0,0,S.W,S.H);drawAnnotated(vctx,1,{});
}
function annotatedCanvas(maxDim,quality){
  const sc=Math.min(1,maxDim/Math.max(S.W,S.H));const c=document.createElement("canvas");
  c.width=Math.round(S.W*sc);c.height=Math.round(S.H*sc);const x=c.getContext("2d");x.imageSmoothingQuality="high";
  const keep={...S.view};S.view.outline=true;S.view.mask=false;buildOutline();drawAnnotated(x,sc,{export:true});Object.assign(S.view,keep);buildOutline();
  return c;
}

function legend(){
}
function included(){return S.parts.filter(p=>!p.excl);}
function renderSummary(){}
const COLS=[["id","ID"],["cls","Class"],["color","Colour"],["len","Length µm"],["wid","Width µm"],["ecd","ECD µm"],["area","Area µm²"],["ar","L:W"],["circ","Circ."],["sol","Solidity"],["conf","Conf."]];
function rowVals(p){const m=um(p);return{id:p.id,cls:CLS[p.cls].name,color:p.color,len:m.len,wid:m.wid,ecd:m.ecd,area:m.area,ar:p.ar,circ:p.circ,sol:p.solidity,conf:p.conf};}
function renderTable(){}

function renderInspector(){}

function refresh(rebuild){
  if(rebuild)buildOutline();
  drawView();
  const now=performance.now();
  if(!S.live||now-lastUi>600){lastUi=now;renderSummary();renderTable();renderInspector();}
  const u=S.umPerPx*S.scale;
  if(S.base){const inc=included(),cnt={};for(const k in CLS)cnt[k]=0;for(const p of inc)cnt[p.cls]++;
    $("#readout").innerHTML=`<div class="total"><b>${inc.length}</b><span>${inc.length===1?"particle":"particles"}</span></div><ul>${Object.entries(CLS).filter(([k])=>Object.keys(CLS).length<=6||cnt[k]>0).map(([k,v])=>`<li><i style="background:${v.color}"></i>${v.name}<b>${cnt[k]}</b></li>`).join("")}</ul>`;
    $("#fps").textContent=S.live&&fps?fps.toFixed(0)+" fps":"";}
  const has=!!S.base&&!S.live;

}

/* ---------- loading images ---------- */
function loadFrom(src,w,h,name,example,maxDim){
  const max=maxDim||1600,sc=Math.min(1,max/Math.max(w,h)),W=Math.max(1,Math.round(w*sc)),H=Math.max(1,Math.round(h*sc));
  const c=document.createElement("canvas");c.width=W;c.height=H;const x=c.getContext("2d",{willReadFrequently:true});x.drawImage(src,0,0,W,H);
  S.base=c;S.W=W;S.H=H;S.rgba=x.getImageData(0,0,W,H).data;S.scale=w/W;S.origW=w;S.origH=h;S.name=name;S.example=example;
  $("#imgName").textContent=name;view.hidden=false;$("#stage").hidden=false;
  return analyze();
}
function loadFile(file){
  if(!file||!/^image\//.test(file.type)){toast("That file isn't an image. Use JPEG, PNG or WebP.");return;}
  const url=URL.createObjectURL(file);const im=new Image();
  im.onload=()=>{stopLive();loadFrom(im,im.naturalWidth,im.naturalHeight,file.name||"Camera photo",false);URL.revokeObjectURL(url);};
  im.onerror=()=>{toast("This image format can't be read here. Export it as JPEG or PNG and try again.");URL.revokeObjectURL(url);};
  im.src=url;
}

/* ---------- interaction ---------- */
function canvasPt(e){const r=view.getBoundingClientRect(),sc=Math.min(r.width/S.W,r.height/S.H),ox=r.left+(r.width-S.W*sc)/2,oy=r.top+(r.height-S.H*sc)/2;return[(e.clientX-ox)/sc,(e.clientY-oy)/sc];}
view.addEventListener("click",e=>{
  if(!S.base)return;const [x,y]=canvasPt(e);
  if(S.cal){if(S.cal.pts.length<2){S.cal.pts.push([x,y]);drawView();if(S.cal.pts.length===2){const [a,b]=S.cal.pts;S.cal.len=Math.hypot(a[0]-b[0],a[1]-b[1]);
    $("#calMsg").textContent=`Line: ${Math.round(S.cal.len*S.scale)} px. Known length:`;["#calLen","#calUnit","#calApply"].forEach(s=>$(s).hidden=false);$("#calLen").focus();}}return;}
  const xi=Math.floor(x),yi=Math.floor(y);let best=-1;
  const r=Math.max(4,Math.round(S.W/200));let bd=Infinity;
  for(let dy=-r;dy<=r;dy++)for(let dx=-r;dx<=r;dx++){const xx=xi+dx,yy=yi+dy;if(xx<0||yy<0||xx>=S.W||yy>=S.H)continue;const L=S.lab[yy*S.W+xx];if(!L)continue;const k=S.labIdx[L];if(k<0)continue;const dd=dx*dx+dy*dy;if(dd<bd){bd=dd;best=k;}}
  select(best,true);
});
function select(i,scroll){S.sel=i;drawView();renderTable();renderInspector();
}



document.addEventListener("keydown",e=>{if(e.target.matches("input,select,textarea"))return;if(!S.parts.length)return;
  if(e.key==="ArrowRight"||e.key==="j"){select((S.sel+1)%S.parts.length,true);e.preventDefault();}
  if(e.key==="ArrowLeft"||e.key==="k"){select((S.sel-1+S.parts.length)%S.parts.length,true);e.preventDefault();}});


// acquisition
$("#btnUpload").onclick=()=>$("#inUpload").click();
$("#inUpload").onchange=e=>{const f=e.target.files;if(f.length>1)toast(`Loaded the first of ${f.length} images. Save it, then upload the next.`);loadFile(f[0]);e.target.value="";};
/* ---------- YOLOv8 model (ONNX Runtime Web) ---------- */
const ORT_VER="1.20.1";
// ONNX Runtime Web is bundled in vendor/ so the app works offline.
const ORT_BASE=new URL("vendor/onnxruntime-web/",location.href).href;
// Default model loaded at startup (change to use another exported YOLO model).
// The YOLO11 model the app uses.
const MODEL_URL="model/yolo11s.onnx";
const NAMED={fiber:"#ff2e88",fibre:"#ff2e88",filament:"#b18cff",film:"#29d3f0",fragment:"#ffb020",granule:"#9be15d",pellet:"#9be15d",bead:"#9be15d",foam:"#ff7a45",microbead:"#9be15d"};
const PALETTE=["#ff2e88","#ffb020","#29d3f0","#9be15d","#b18cff","#ff7a45","#4dabf7","#f783ac","#ffd43b","#69db7c","#e599f7","#63e6be"];
// conf: minimum confidence for a detection (0.25 = YOLO default). iou: overlap for merging duplicates.
const M={ready:null,session:null,name:"",task:"detect",names:[],imgsz:640,conf:0.25,iou:0.45,tok:0,lb:null};
function ensureOrt(){
  if(window.ort)return Promise.resolve();
  return new Promise((res,rej)=>{const sc=document.createElement("script");sc.src=ORT_BASE+"ort.wasm.min.js";sc.onload=()=>{
      ort.env.wasm.numThreads=1;ort.env.wasm.wasmPaths=ORT_BASE;res();};
    sc.onerror=()=>rej(new Error("The model runtime couldn't be loaded. Check that vendor/onnxruntime-web/ is next to index.html and that the app is opened through a local server (npm start)."));document.head.appendChild(sc);});
}
function onnxMeta(buf){
  const u8=new Uint8Array(buf),out={},dec=new TextDecoder(),enc=new TextEncoder();
  for(const key of ["names","imgsz","task","description"]){const kb=enc.encode(key),pat=[0x0a,kb.length,...kb,0x12];
    outer:for(let i=u8.length-pat.length-2;i>=0;i--){if(u8[i]!==0x0a)continue;for(let j=1;j<pat.length;j++)if(u8[i+j]!==pat[j])continue outer;
      let k=i+pat.length,len=0,sh=0,b;do{b=u8[k++];len+=(b&127)*2**sh;sh+=7;}while(b&128&&sh<35);
      if(len>0&&k+len<=u8.length){out[key]=dec.decode(u8.subarray(k,k+len));break;}}}
  return out;
}
function parseNames(str){const names=[];if(!str)return names;for(const m of str.matchAll(/(\d+)\s*:\s*(['"])(.*?)\2/g))names[+m[1]]=m[3];return names;}
function buildModelClasses(){
  const c={};const used=new Set();
  M.names.forEach((n,i)=>{const key="m"+i,low=String(n).toLowerCase().trim();let col=NAMED[low]||Object.entries(NAMED).find(([k])=>low.includes(k))?.[1];
    if(!col||used.has(col))col=PALETTE.find(x=>!used.has(x))||PALETTE[i%PALETTE.length];used.add(col);
    c[key]={name:String(n).replace(/^./,ch=>ch.toUpperCase()),color:col,fiber:/fib|filament/.test(low)};});
  CLS=c;
}
function showMsg(html){const m=$("#liveMsg");m.innerHTML=html;m.hidden=!html;}
async function loadModel(url){
  showMsg("");
  try{
    await ensureOrt();
    const r=await fetch(url);
    if(!r.ok)throw new Error(`${url} was not found. Check that the file is in the model folder.`);
    const buf=await r.arrayBuffer();
    const meta=onnxMeta(buf);
    const sess=await ort.InferenceSession.create(new Uint8Array(buf),{executionProviders:["wasm"],graphOptimizationLevel:"all"});
    const isz=(meta.imgsz||"").match(/\d+/g);const imgsz=isz?+isz[0]:640;
    const dummy=new ort.Tensor("float32",new Float32Array(3*imgsz*imgsz),[1,3,imgsz,imgsz]);
    const out=await sess.run({[sess.inputNames[0]]:dummy});
    const o0=out[sess.outputNames[0]];
    if(!o0||o0.dims.length!==3)throw new Error("The model file isn't a standard YOLO11 ONNX export.");
    const seg=sess.outputNames.length>1&&out[sess.outputNames[1]]?.dims.length===4;
    const nc=o0.dims[1]-4-(seg?32:0);
    let names=parseNames(meta.names);if(names.length!==nc)names=Array.from({length:nc},(_,i)=>names[i]||"class "+i);
    Object.assign(M,{session:sess,name:url.split("/").pop(),task:seg?"segment":"detect",names,imgsz});
    buildModelClasses();
  }catch(e){
    M.session=null;
    const local=location.protocol==="file:";
    let why=esc(e?.message||String(e));if(!/[.!?]$/.test(why))why+=".";
    showMsg("The YOLO11 model couldn't be loaded. "+why+(local?" Open the app through the local server (run <code>npm start</code>, then go to http://localhost:5173) instead of opening index.html directly.":""));
  }
}
// Load the model once at startup; analysis waits for it.
M.ready=loadModel(MODEL_URL);
// Remove a model remembered by older versions of this app.
try{indexedDB.deleteDatabase("microplastic-bench");}catch(e){}

async function analyzeYolo(){
  const tok=++M.tok,t0=performance.now();
  const {W,H}=S,sz=M.imgsz,r=Math.min(sz/W,sz/H),nw=Math.round(W*r),nh=Math.round(H*r);
  const padL=Math.round((sz-nw)/2-0.1),padT=Math.round((sz-nh)/2-0.1);
  const c=M.lb||(M.lb=document.createElement("canvas"));c.width=sz;c.height=sz;
  const x=c.getContext("2d",{willReadFrequently:true});x.fillStyle="rgb(114,114,114)";x.fillRect(0,0,sz,sz);x.drawImage(S.base,padL,padT,nw,nh);
  const px=x.getImageData(0,0,sz,sz).data,N=sz*sz,f=new Float32Array(3*N);
  for(let i=0,j=0;i<N;i++,j+=4){f[i]=px[j]/255;f[i+N]=px[j+1]/255;f[i+2*N]=px[j+2]/255;}
  const sess=M.session;
  const out=await sess.run({[sess.inputNames[0]]:new ort.Tensor("float32",f,[1,3,sz,sz])});
  if(tok!==M.tok||sess!==M.session)return;
  const o0=out[sess.outputNames[0]],C=o0.dims[1],A=o0.dims[2],d=o0.data,seg=M.task==="segment",nm=seg?32:0,nc=C-4-nm;
  // candidates above threshold
  const cand=[];
  for(let a=0;a<A;a++){let best=-1,bs=M.conf;for(let k=0;k<nc;k++){const v=d[(4+k)*A+a];if(v>bs){bs=v;best=k;}}
    if(best<0)continue;const cx=d[a],cy=d[A+a],w=d[2*A+a],h=d[3*A+a];cand.push({a,k:best,s:bs,x1:cx-w/2,y1:cy-h/2,x2:cx+w/2,y2:cy+h/2});}
  cand.sort((p,q)=>q.s-p.s);
  const iou=(p,q)=>{const w=Math.max(0,Math.min(p.x2,q.x2)-Math.max(p.x1,q.x1)),h=Math.max(0,Math.min(p.y2,q.y2)-Math.max(p.y1,q.y1)),i=w*h;return i/((p.x2-p.x1)*(p.y2-p.y1)+(q.x2-q.x1)*(q.y2-q.y1)-i+1e-9);};
  const keep=[];for(const cd of cand){if(keep.length>=300)break;if(keep.every(k=>k.k!==cd.k||iou(k,cd)<=M.iou))keep.push(cd);}
  // paint label map in image coordinates
  const lab=new Int32Array(W*H);
  let protos=null,mh=0,mw=0;if(seg){const o1=out[sess.outputNames[1]];protos=o1.data;mh=o1.dims[2];mw=o1.dims[3];}
  keep.forEach((b,i)=>{
    const L=i+1;
    const X1=clamp(Math.floor((b.x1-padL)/r),0,W-1),Y1=clamp(Math.floor((b.y1-padT)/r),0,H-1),X2=clamp(Math.ceil((b.x2-padL)/r),0,W),Y2=clamp(Math.ceil((b.y2-padT)/r),0,H);
    if(X2<=X1||Y2<=Y1)return;
    if(!seg){for(let y=Y1;y<Y2;y++)for(let xx=X1;xx<X2;xx++){const p=y*W+xx;if(!lab[p])lab[p]=L;}return;}
    // mask at prototype resolution inside the box
    const co=new Float32Array(32);for(let m=0;m<32;m++)co[m]=d[(4+nc+m)*A+b.a];
    const sx=mw/sz,sy=mh/sz;
    const mx1=clamp(Math.floor(b.x1*sx),0,mw-1),my1=clamp(Math.floor(b.y1*sy),0,mh-1),mx2=clamp(Math.ceil(b.x2*sx),1,mw),my2=clamp(Math.ceil(b.y2*sy),1,mh);
    const bw=mx2-mx1,bh=my2-my1,mk=new Float32Array(bw*bh),PS=mh*mw;
    for(let yy=0;yy<bh;yy++)for(let xx=0;xx<bw;xx++){const pi=(my1+yy)*mw+(mx1+xx);let v=0;for(let m=0;m<32;m++)v+=co[m]*protos[m*PS+pi];mk[yy*bw+xx]=v;}
    for(let y=Y1;y<Y2;y++){const ly=(y+0.5)*r+padT;if(ly<b.y1||ly>b.y2)continue;
      const fy=ly*sy-0.5-my1,y0=clamp(Math.floor(fy),0,bh-1),y1=Math.min(y0+1,bh-1),ty=clamp(fy-y0,0,1);
      for(let xx=X1;xx<X2;xx++){const lx=(xx+0.5)*r+padL;if(lx<b.x1||lx>b.x2)continue;
        const p=y*W+xx;if(lab[p])continue;
        const fx=lx*sx-0.5-mx1,x0=clamp(Math.floor(fx),0,bw-1),x1=Math.min(x0+1,bw-1),tx=clamp(fx-x0,0,1);
        const v=(mk[y0*bw+x0]*(1-tx)+mk[y0*bw+x1]*tx)*(1-ty)+(mk[y1*bw+x0]*(1-tx)+mk[y1*bw+x1]*tx)*ty;
        if(v>0)lab[p]=L;}}
  });
  // group pixels per label
  const nL=keep.length,cnt=new Int32Array(nL+2);for(let i=0;i<lab.length;i++)if(lab[i])cnt[lab[i]]++;
  const start=new Int32Array(nL+2);for(let L=1;L<=nL+1;L++)start[L]=start[L-1]+cnt[L-1];
  const q=new Int32Array(start[nL]+cnt[nL]),fillp=start.slice();for(let i=0;i<lab.length;i++){const L=lab[i];if(L)q[fillp[L]++]=i;}
  const dz=new Uint8Array(W*H),umPx=S.umPerPx*S.scale,parts=[];
  for(let L=1;L<=nL;L++){if(!cnt[L])continue;const b=keep[L-1];
    const p=features({L,start:start[L],end:start[L]+cnt[L],maxd:0},lab,q,dz,umPx);p.modelCls="m"+b.k;p.score=b.s;p.color=p.color;parts.push(p);}
  const labIdx=new Int32Array(nL+1).fill(-1);
  const band=Math.max(40,H/8);
  parts.sort((a,b)=>(Math.floor(a.cy/band)-Math.floor(b.cy/band))||(a.cx-b.cx));
  parts.forEach((p,i)=>{p.id=i+1;labIdx[p.L]=i;});
  S.d=dz;S.lab=lab;S.labIdx=labIdx;S.parts=parts;S.sel=-1;S.T={hi:0,lo:0};
  classifyAll();S.ms=Math.round(performance.now()-t0);refresh(true);
}

/* ---------- live detection ---------- */
let stream=null,liveTimer=null,lastUi=0,fpsT=0,fpsN=0,fps=0;
async function startLive(deviceId){
  const msg=$("#liveMsg");msg.hidden=true;
  try{
    if(!navigator.mediaDevices?.getUserMedia)throw new Error("no api");
    if(stream)stream.getTracks().forEach(t=>t.stop());
    stream=await navigator.mediaDevices.getUserMedia({audio:false,video:deviceId?{deviceId:{exact:deviceId},width:{ideal:1920},height:{ideal:1080}}:{facingMode:"environment",width:{ideal:1920},height:{ideal:1080}}});
    const v=$("#video");v.srcObject=stream;await v.play();
    S.live=true;S.example=false;$("#liveTag").hidden=false;$("#liveCtl").hidden=false;$("#btnLive").classList.replace("primary","stop");$("#btnLive").innerHTML=STOP_BTN;
    try{const devs=(await navigator.mediaDevices.enumerateDevices()).filter(d=>d.kind==="videoinput");const sel=$("#camSel");const cur=stream.getVideoTracks()[0]?.getSettings().deviceId;
      sel.innerHTML=devs.map((d,i)=>`<option value="${esc(d.deviceId)}"${d.deviceId===cur?" selected":""}>${esc(d.label||"Camera "+(i+1))}</option>`).join("");$("#camPick").hidden=devs.length<2;}catch(e){}
    clearTimeout(liveTimer);liveTick();
  }catch(err){
    S.live=false;
    msg.textContent="Camera access was refused or no camera was found. Allow camera access for this page in your browser settings, then try again. The camera only works on http://localhost or https:// addresses.";
    msg.hidden=false;
  }
}
async function liveTick(){
  if(!S.live)return;const v=$("#video");const t0=performance.now();
  if(v.videoWidth){try{await loadFrom(v,v.videoWidth,v.videoHeight,"Live camera",false,960);}catch(e){}
    if(!S.live)return;
    fpsN++;if(t0-fpsT>1000){fps=fpsN*1000/(t0-fpsT);fpsN=0;fpsT=t0;}}
  const took=performance.now()-t0;liveTimer=setTimeout(liveTick,Math.max(30,took*0.5));
}
function stopLive(){
  if(!S.live&&!stream)return;S.live=false;clearTimeout(liveTimer);
  if(stream){stream.getTracks().forEach(t=>t.stop());stream=null;}
  $("#liveTag").hidden=true;$("#liveCtl").hidden=true;$("#camPick").hidden=true;$("#btnLive").classList.replace("stop","primary");$("#btnLive").innerHTML=LIVE_BTN;refresh(false);
}
const LIVE_BTN=$("#btnLive").innerHTML;
const STOP_BTN='<svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="1"/></svg><span>Stop</span>';
$("#btnLive").onclick=()=>{if(S.live)stopLive();else startLive();};
$("#camSel").onchange=e=>startLive(e.target.value);
$("#btnFreeze").onclick=()=>{const v=$("#video");if(!v.videoWidth)return;const c=document.createElement("canvas");c.width=v.videoWidth;c.height=v.videoHeight;c.getContext("2d").drawImage(v,0,0);
  stopLive();loadFrom(c,c.width,c.height,"Capture "+new Date().toLocaleString(),false);toast("Frame frozen");};
const wrap=document;
["dragenter","dragover"].forEach(t=>wrap.addEventListener(t,e=>{e.preventDefault();$("#drop").hidden=false;}));
["dragleave","drop"].forEach(t=>wrap.addEventListener(t,e=>{e.preventDefault();$("#drop").hidden=true;}));
wrap.addEventListener("drop",e=>{const f=e.dataTransfer?.files?.[0];if(f)loadFile(f);});
document.addEventListener("paste",e=>{const it=[...(e.clipboardData?.items||[])].find(i=>i.type.startsWith("image/"));if(it){loadFile(it.getAsFile());e.preventDefault();}});

// settings
let tmr=null;const reanalyze=()=>{clearTimeout(tmr);tmr=setTimeout(()=>{savePrefs();analyze();},120);};
function bindSettings(){}
const endCal=()=>{S.cal=null;$("#calbar").hidden=true;drawView();};
$("#calCancel").onclick=endCal;

/* ---------- export ---------- */
function meta(){return{sampleId:(S.name||"image").replace(/\.[^.]+$/,"").slice(0,60)};}
function particleRecords(){
  return S.parts.map(p=>{const m=um(p);return{id:p.id,cls:p.cls,src:p.override?"manual":"auto",conf:+p.conf.toFixed(2),color:p.color,
    len:+m.len.toFixed(1),wid:+m.wid.toFixed(1),fmax:+m.fmax.toFixed(1),fmin:+m.fmin.toFixed(1),ecd:+m.ecd.toFixed(1),area:+m.area.toFixed(0),
    ar:+p.ar.toFixed(3),circ:+p.circ.toFixed(3),sol:+p.solidity.toFixed(3),round:+p.roundness.toFixed(3),holes:+p.holeFrac.toFixed(3),
    x:Math.round(p.cx*S.scale),y:Math.round(p.cy*S.scale),edge:p.edge};});
}
const csvEsc=v=>{const s=v==null?"":String(v);return/[",\n]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s;};
function particlesCsv(sampleId,recs){
  const head=["image","particle_id","morphology","class_source","confidence","colour","length_um","width_um","feret_max_um","feret_min_um","ecd_um","area_um2","aspect_ratio","circularity","solidity","roundness","hole_fraction","centroid_x_px","centroid_y_px","edge_touching"];
  return[head.join(",")].concat(recs.map(r=>[sampleId,r.id,r.cls,r.src,r.conf,r.color,r.len,r.wid,r.fmax,r.fmin,r.ecd,r.area,r.ar,r.circ,r.sol,r.round,r.holes,r.x,r.y,r.edge].map(csvEsc).join(","))).join("\n");
}
const slug=s=>(s||"sample").replace(/[^\w.-]+/g,"_").slice(0,60);
function offerFile(filename,data){
  const blob=data instanceof Blob?data:new Blob([data],{type:"text/plain"});const a=document.createElement("a");
  a.href=URL.createObjectURL(blob);a.download=filename;document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(a.href),5000);
}

const esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
/* ---------- capabilities ---------- */
function initCaps(){}
/* ---------- misc ---------- */
let toastT=null;function toast(t){let el=$(".toast");if(!el){el=document.createElement("div");el.className="toast";el.setAttribute("role","status");document.body.appendChild(el);}el.textContent=t;el.hidden=false;clearTimeout(toastT);toastT=setTimeout(()=>el.hidden=true,3800);}

legend();refresh(false);
initCaps();
})();
