/* ============================================================
   研招网报名照自测
   全部检测在浏览器本地完成，照片不会被上传。
   依据：《网上确认图像采集标准》
   ============================================================ */

import { FaceLandmarker, FilesetResolver } from './vision_bundle.mjs';

/* ---------- 参考线几何（取自研招网 bg-line.png 实测） ---------- */
const GUIDE = {
  vx: [0.250, 0.749],   // 两条竖线：画面宽度 25% / 75%
  hy: 0.390,            // 横线：画面高度 39%（官方 30%~50% 区间中值）
};
const EYE_BAND = [0.30, 0.50];  // 官方允许的眼睛高度区间

/* ---------- 判定阈值 ---------- */
const T = {
  ratio:      { target: 0.75, tol: 0.01 },   // 宽高比 3:4
  minPx:      { w: 480, h: 640 },            // 建议最小分辨率
  maxBytes:   10 * 1024 * 1024,              // 10 MB
  eyeY:       EYE_BAND,
  centerTol:  0.05,                          // 水平居中容差 ±5%
  centerWarn: 0.08,
  angle:      10,                            // 官方上限 ±10°
  angleWarn:  13,                            // 模型估算存在误差，10°~13° 提示人工复核
  personRatio: 2 / 3,                        // 头肩占比 ≥ 2/3
  personWarn:  0.55,
  brightness: [80, 200],                     // 平均亮度合理区间
  overexpose: 0.03,                          // 过曝像素比例上限
  sharpness:  100,                           // 拉普拉斯方差下限
};

/** 角度判定：≤10° 通过；10~13° 提示复核；>13° 不合格 */
const angleStatus = (degVal) => {
  const a = Math.abs(degVal);
  return a <= T.angle ? 'pass' : a <= T.angleWarn ? 'warn' : 'fail';
};

/* ---------- 关键点索引（MediaPipe FaceMesh 478 点） ---------- */
const LM = {
  irisA: [468, 469, 470, 471, 472],   // 一侧虹膜
  irisB: [473, 474, 475, 476, 477],   // 另一侧虹膜
  eyeA:  [33, 133, 159, 145],         // 一侧眼廓
  eyeB:  [362, 263, 386, 374],        // 另一侧眼廓
  faceOval: [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
  chin: 152, forehead: 10, noseTip: 1,
};

/* ============================================================
   工具函数
   ============================================================ */
const $ = (id) => document.getElementById(id);
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const deg = (r) => (r * 180) / Math.PI;
const pct = (v, d = 1) => `${(v * 100).toFixed(d)}%`;
const fmtSize = (b) =>
  b < 1024 * 1024 ? `${(b / 1024).toFixed(0)} KB` : `${(b / 1048576).toFixed(2)} MB`;
const mean = (arr) => arr.reduce((s, v) => s + v, 0) / arr.length;

/* ============================================================
   模型加载
   ============================================================ */
let landmarker = null;
let modelState = 'idle'; // idle | loading | ready | error
let modelPromise = null;

function loadModel() {
  if (modelPromise) return modelPromise;
  modelState = 'loading';
  modelPromise = (async () => {
    const fileset = await FilesetResolver.forVisionTasks('./assets/wasm');
    const opts = (delegate) => ({
      baseOptions: { modelAssetPath: './assets/face_landmarker.task', delegate },
      runningMode: 'IMAGE',
      numFaces: 2,
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: true,
    });
    try {
      landmarker = await FaceLandmarker.createFromOptions(fileset, opts('GPU'));
    } catch (e) {
      console.warn('GPU delegate 不可用，回退 CPU：', e);
      landmarker = await FaceLandmarker.createFromOptions(fileset, opts('CPU'));
    }
    modelState = 'ready';
    return landmarker;
  })().catch((e) => {
    modelState = 'error';
    console.error(e);
    throw e;
  });
  return modelPromise;
}

/* ============================================================
   像素级分析：背景、亮度、过曝、清晰度、人像占比
   ============================================================ */
function analyzePixels(img) {
  const MAX = 640;
  const scale = Math.min(1, MAX / Math.max(img.naturalWidth, img.naturalHeight));
  const w = Math.max(1, Math.round(img.naturalWidth * scale));
  const h = Math.max(1, Math.round(img.naturalHeight * scale));

  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0, w, h);
  const { data } = ctx.getImageData(0, 0, w, h);

  const N = w * h;
  const lum = new Float32Array(N);
  let sum = 0;
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    const y = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    lum[p] = y; sum += y;
  }
  const brightness = sum / N;

  /* --- 背景色：取上方左、中、右及两侧区域的中位数 --- */
  const patches = [
    [0, 0, 0.14, 0.07], [0.43, 0, 0.14, 0.035], [0.86, 0, 0.14, 0.07],
    [0, 0.10, 0.06, 0.10], [0.94, 0.10, 0.06, 0.10],
  ];
  const samples = [];
  for (const [fx, fy, fw, fh] of patches) {
    const x0 = Math.floor(fx * w), y0 = Math.floor(fy * h);
    const x1 = Math.min(w, Math.ceil((fx + fw) * w)), y1 = Math.min(h, Math.ceil((fy + fh) * h));
    for (let y = y0; y < y1; y += 2)
      for (let x = x0; x < x1; x += 2) {
        const i = (y * w + x) * 4;
        samples.push([data[i], data[i + 1], data[i + 2]]);
      }
  }
  const bg = [0, 1, 2].map((k) => {
    const v = samples.map((s) => s[k]).sort((a, b) => a - b);
    return v[Math.floor(v.length / 2)];
  });

  /* --- 背景分类 --- */
  const [br, bgc, bb] = bg;
  const mx = Math.max(br, bgc, bb), mn = Math.min(br, bgc, bb);
  let bgType = 'other';
  if (mn > 200 && mx - mn < 24) bgType = 'white';
  else if (bb > br + 22 && bb > 105 && bb >= bgc - 12) bgType = 'blue';

  /* --- 从画面四边泛洪，把与背景连通的区域判为背景 ---
         这样白色衬衫、浅色皮肤不会被误算成背景，反之亦然 --- */
  const TOL = 38, TOL2 = TOL * TOL;
  const bgMask = new Uint8Array(N);
  const stack = new Int32Array(N);
  let sp = 0;
  const tryPush = (p) => {
    if (bgMask[p]) return;
    const i = p * 4;
    const dr = data[i] - br, dg = data[i + 1] - bgc, db = data[i + 2] - bb;
    if (dr * dr + dg * dg + db * db <= TOL2) { bgMask[p] = 1; stack[sp++] = p; }
  };
  for (let x = 0; x < w; x++) { tryPush(x); tryPush((h - 1) * w + x); }
  for (let y = 0; y < h; y++) { tryPush(y * w); tryPush(y * w + w - 1); }
  while (sp > 0) {
    const p = stack[--sp];
    const x = p % w, y = (p / w) | 0;
    if (x > 0) tryPush(p - 1);
    if (x < w - 1) tryPush(p + 1);
    if (y > 0) tryPush(p - w);
    if (y < h - 1) tryPush(p + w);
  }

  /* --- 人像统计：面积占比、顶部位置、过曝 --- */
  let person = 0, over = 0, personTop = h;
  for (let p = 0; p < N; p++) {
    if (bgMask[p]) continue;
    person++;
    if (lum[p] > 250) over++;
    const y = (p / w) | 0;
    if (y < personTop) personTop = y;
  }
  const personRatio = person / N;
  const overRatio = person > 0 ? over / person : 0;
  const personTopRatio = personTop / h;

  /* --- 清晰度：先把亮度对比度归一化，再做拉普拉斯方差，
         避免「照片偏暗」被误判成「照片模糊」 --- */
  const sorted = Float32Array.from(lum).sort();
  const lo = sorted[Math.floor(N * 0.01)];
  const hi = sorted[Math.floor(N * 0.99)];
  const span = Math.max(1, hi - lo);
  const lumN = new Float32Array(N);
  for (let p = 0; p < N; p++) lumN[p] = ((lum[p] - lo) / span) * 255;

  let lapSum = 0, lapSq = 0, n = 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const p = y * w + x;
      const v = 4 * lumN[p] - lumN[p - 1] - lumN[p + 1] - lumN[p - w] - lumN[p + w];
      lapSum += v; lapSq += v * v; n++;
    }
  }
  const lapMean = lapSum / n;
  const sharpness = lapSq / n - lapMean * lapMean;

  return { bg, bgType, brightness, overRatio, sharpness, personRatio, personTopRatio };
}

/* ============================================================
   人脸关键点 → 姿态
   ============================================================ */
function centroid(pts, idxs) {
  let x = 0, y = 0, z = 0;
  for (const i of idxs) { x += pts[i].x; y += pts[i].y; z += pts[i].z || 0; }
  const k = idxs.length;
  return { x: x / k, y: y / k, z: z / k };
}

/** 由 facialTransformationMatrix（列主序 16 元素）解出三轴角度 */
function eulerFromMatrix(d) {
  const r00 = d[0], r10 = d[1], r20 = d[2];
  const r11 = d[5], r21 = d[6];
  const r22 = d[10];
  const pitch = Math.atan2(-r21, r22);
  const yaw = Math.atan2(r20, Math.sqrt(r21 * r21 + r22 * r22));
  const roll = Math.atan2(-r10, r00);
  return { pitch: deg(pitch), yaw: deg(yaw), roll: deg(roll) };
}

function analyzeFace(result, img) {
  const faces = result.faceLandmarks || [];
  if (!faces.length) return { count: 0 };

  const lm = faces[0];
  const W = img.naturalWidth, H = img.naturalHeight;

  // 虹膜中心（更接近“瞳孔”），按 x 排序区分左右
  const irises = [centroid(lm, LM.irisA), centroid(lm, LM.irisB)].sort((a, b) => a.x - b.x);
  const [eyeL, eyeR] = irises;
  const eyeMid = { x: (eyeL.x + eyeR.x) / 2, y: (eyeL.y + eyeR.y) / 2 };
  const eyeDist = Math.hypot(eyeR.x - eyeL.x, eyeR.y - eyeL.y);
  const eyeLineRoll = deg(Math.atan2(eyeR.y - eyeL.y, eyeR.x - eyeL.x));

  // 面部框（用脸廓关键点）
  const oval = LM.faceOval.map((i) => lm[i]);
  const xs = oval.map((p) => p.x), ys = oval.map((p) => p.y);
  const box = {
    x0: Math.min(...xs), x1: Math.max(...xs),
    y0: Math.min(...ys), y1: Math.max(...ys),
  };
  box.cx = (box.x0 + box.x1) / 2;
  box.cy = (box.y0 + box.y1) / 2;

  // 三轴角度
  let pose = null;
  const mats = result.facialTransformationMatrixes || [];
  if (mats.length && mats[0].data && mats[0].data.length >= 16) {
    pose = eulerFromMatrix(mats[0].data);
  }

  // 表情 / 睁眼
  let blink = null, jawOpen = null;
  const bs = result.faceBlendshapes || [];
  if (bs.length && bs[0].categories) {
    const m = {};
    for (const c of bs[0].categories) m[c.categoryName] = c.score;
    if (m.eyeBlinkLeft != null && m.eyeBlinkRight != null)
      blink = Math.max(m.eyeBlinkLeft, m.eyeBlinkRight);
    if (m.jawOpen != null) jawOpen = m.jawOpen;
  }

  return {
    count: faces.length,
    eyeL, eyeR, eyeMid, eyeDist, eyeLineRoll,
    box, pose, blink, jawOpen, lm,
    eyeY: eyeMid.y,          // 归一化（0~1）
    eyeX: eyeMid.x,
    faceCx: box.cx,
    faceCy: box.cy,
    W, H,
  };
}

/* ============================================================
   检测项
   ============================================================ */
function buildChecks(file, img, px, face) {
  const out = [];
  const add = (group, name, req, value, status, hint) =>
    out.push({ group, name, req, value, status, hint });

  /* ---------- 一、文件与规格 ---------- */
  const ext = (file.name.split('.').pop() || '').toLowerCase();
  const isJpeg = file.type === 'image/jpeg' || ext === 'jpg' || ext === 'jpeg';
  add('文件与规格', '图片格式', '仅支持 jpg / jpeg',
    isJpeg ? 'JPG' : ext.toUpperCase() || '未知', isJpeg ? 'pass' : 'fail',
    isJpeg ? null : '当前格式不被受理，请另存为 JPG 后重新上传。');

  const sizeOk = file.size <= T.maxBytes;
  add('文件与规格', '文件大小', '建议不超过 10 MB',
    fmtSize(file.size), sizeOk ? 'pass' : 'warn',
    sizeOk ? null : '文件偏大，上传可能超时，建议压缩到 10 MB 以内。');

  const ratio = img.naturalWidth / img.naturalHeight;
  const ratioOk = Math.abs(ratio - T.ratio.target) <= T.ratio.tol;
  add('文件与规格', '宽高比例', '严格 3:4（即 0.750）',
    ratio.toFixed(3), ratioOk ? 'pass' : 'fail',
    ratioOk ? null : `当前约 ${ratio.toFixed(2)}:1，非 3:4。直接上传会被系统拉伸变形，请先按 3:4 裁剪。`);

  const pxOk = img.naturalWidth >= T.minPx.w && img.naturalHeight >= T.minPx.h;
  add('文件与规格', '分辨率', '建议不低于 480×640',
    `${img.naturalWidth}×${img.naturalHeight}`, pxOk ? 'pass' : 'warn',
    pxOk ? null : '分辨率偏低，打印或人脸核验时可能不够清晰。');

  /* ---------- 二、人像姿态 ---------- */
  if (!face || face.count === 0) {
    add('人像姿态', '人脸检出', '须检出且仅检出 1 张正脸', '未检出', 'fail',
      '未能在照片中定位到人脸。请确认照片为正面、清晰、无遮挡的半身证件照。');
    for (const n of ['眼睛高度', '人像水平居中', '倾斜角', '水平转动角', '俯仰角', '头部与肩部占比', '双眼睁开', '嘴唇闭合'])
      add('人像姿态', n, '—', '无法判定', 'skip', null);
  } else {
    const multi = face.count > 1;
    add('人像姿态', '人脸检出', '须检出且仅检出 1 张正脸',
      multi ? `检出 ${face.count} 张` : '检出 1 张', multi ? 'fail' : 'pass',
      multi ? '画面中出现了多张人脸，请使用只有本人的单人照片。' : null);

    // 眼睛高度
    const y = face.eyeY;
    const yOk = y >= T.eyeY[0] && y <= T.eyeY[1];
    add('人像姿态', '眼睛高度', '瞳孔距照片上边沿为图像高度的 30%–50%',
      pct(y), yOk ? 'pass' : 'fail',
      yOk ? null
          : y < T.eyeY[0]
            ? `眼睛偏高（应 ≥30%），头顶留白不足。参考线在 ${pct(GUIDE.hy)}，建议向下留出更多空间。`
            : `眼睛偏低（应 ≤50%），头顶留白过多。参考线在 ${pct(GUIDE.hy)}，建议裁掉顶部多余空白。`);

    // 水平居中
    const off = Math.abs(face.faceCx - 0.5);
    const cStatus = off <= T.centerTol ? 'pass' : off <= T.centerWarn ? 'warn' : 'fail';
    add('人像姿态', '人像水平居中', '人像水平居中、左右对称',
      `中心 ${pct(face.faceCx)}（偏移 ${(off * 100).toFixed(1)}%）`, cStatus,
      cStatus === 'pass' ? null
        : `面部中心偏离画面中线 ${(off * 100).toFixed(1)}%，两条竖线（25% / 75%）即为居中参考。`);

    // 三轴角度（由三维模型估算；倾斜角取旋转矩阵的 roll，比眼线估算更准）
    const ANGLE_REQ = '应在 ±10° 之内';
    const roll = face.pose ? face.pose.roll : face.eyeLineRoll;
    const rollS = angleStatus(roll);
    add('人像姿态', '倾斜角（左右歪头）', ANGLE_REQ,
      `${roll.toFixed(1)}°`, rollS,
      rollS === 'pass' ? null : '头部左右歪斜，请把画面摆正或重新拍摄。');

    if (face.pose) {
      const { yaw, pitch } = face.pose;
      const yS = angleStatus(yaw), pS = angleStatus(pitch);
      add('人像姿态', '水平转动角（左右转头）', ANGLE_REQ,
        `${yaw.toFixed(1)}°`, yS,
        yS === 'pass' ? null : '面部朝向偏左或偏右，请正面正对镜头。');
      add('人像姿态', '俯仰角（抬头/低头）', ANGLE_REQ,
        `${pitch.toFixed(1)}°`, pS,
        pS === 'pass' ? null : '头部有明显仰起或低下，请平视镜头。');
    } else {
      add('人像姿态', '水平转动角（左右转头）', ANGLE_REQ, '无法判定', 'skip');
      add('人像姿态', '俯仰角（抬头/低头）', ANGLE_REQ, '无法判定', 'skip');
    }

    // 头肩占比
    const pr = px.personRatio;
    const heightRatio = 1 - px.personTopRatio;
    const pStatus = pr >= T.personRatio ? 'pass' : pr >= T.personWarn ? 'warn' : 'fail';
    add('人像姿态', '头部与肩部占比', '人像占整张照片的比例不小于 2/3',
      `面积 ${pct(pr, 0)}`, pStatus,
      pStatus === 'pass' ? null
        : `人像面积约 ${pct(pr, 0)}，低于 2/3（人像纵向占画面 ${pct(heightRatio, 0)}）。` +
          '可适当靠近拍摄或裁掉两侧多余背景，使人像更饱满。');

    // 睁眼 / 嘴部
    if (face.blink != null) {
      const open = face.blink < 0.5;
      add('人像姿态', '双眼睁开', '双眼自然睁开并平视',
        open ? '睁开' : `闭眼倾向 ${pct(face.blink, 0)}`, open ? 'pass' : 'fail',
        open ? null : '检测到眼睛接近闭合，请使用双眼自然睁开的照片。');
    } else {
      add('人像姿态', '双眼睁开', '双眼自然睁开并平视', '无法判定', 'skip');
    }
    if (face.jawOpen != null) {
      const closed = face.jawOpen < 0.35;
      add('人像姿态', '嘴唇自然闭合', '嘴唇自然闭合、表情平和',
        closed ? '闭合' : `张口倾向 ${pct(face.jawOpen, 0)}`, closed ? 'pass' : 'warn',
        closed ? null : '检测到嘴部张开，建议使用嘴唇自然闭合的照片。');
    }
  }

  /* ---------- 三、背景与画质 ---------- */
  const bgLabel = { white: '白色', blue: '蓝色', other: '其他' }[px.bgType];
  const bgOk = px.bgType === 'white' || px.bgType === 'blue';
  add('背景与画质', '背景底色', '白色或蓝色背景（以报考点要求为准）',
    `${bgLabel} rgb(${px.bg.join(',')})`, bgOk ? 'pass' : 'fail',
    bgOk ? null : '未识别到白底或蓝底。若报考点允许，请更换为纯白或纯蓝背景。');

  const [bLo, bHi] = T.brightness;
  const bOk = px.brightness >= bLo && px.brightness <= bHi;
  add('背景与画质', '整体亮度', '照明光线均匀，不过暗也不过曝',
    px.brightness.toFixed(0), bOk ? 'pass' : 'warn',
    bOk ? null : px.brightness < bLo ? '整体偏暗，建议在更明亮均匀的光线下重拍。'
                                     : '整体偏亮，注意避免过曝。');

  const oOk = px.overRatio <= T.overexpose;
  add('背景与画质', '过曝情况', '无高光、光斑',
    pct(px.overRatio, 2), oOk ? 'pass' : 'warn',
    oOk ? null : '画面中纯白像素偏多，可能存在过曝或反光。');

  const sOk = px.sharpness >= T.sharpness;
  add('背景与画质', '清晰度', '人像对焦准确、层次清晰、不模糊',
    px.sharpness.toFixed(0), sOk ? 'pass' : 'fail',
    sOk ? null : '画面偏模糊或有重影，请使用对焦准确、清晰的照片。');

  return out;
}

/* ============================================================
   绘制参考线与关键点
   ============================================================ */
function drawOverlay(img, face, opts) {
  const cv = $('overlayCanvas');
  const W = img.naturalWidth, H = img.naturalHeight;
  cv.width = W; cv.height = H;
  const ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, W, H);

  const lw = Math.max(1, Math.round(W / 480 * 1.5));
  const f = Math.max(11, Math.round(W / 480 * 15));
  ctx.font = `600 ${f}px -apple-system,"PingFang SC",sans-serif`;
  ctx.textBaseline = 'middle';

  if (opts.guides) {
    const [vx1, vx2] = GUIDE.vx.map((v) => v * W);
    const hy = GUIDE.hy * H;

    // 允许区间底纹 30%~50%
    ctx.fillStyle = 'rgba(26,239,243,.07)';
    ctx.fillRect(0, EYE_BAND[0] * H, W, (EYE_BAND[1] - EYE_BAND[0]) * H);

    ctx.strokeStyle = 'rgba(26,239,243,.95)';
    ctx.lineWidth = lw;

    // 两条竖线（贯穿全高）
    for (const x of [vx1, vx2]) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
    }
    // 横线：只画在竖线外侧，不穿过面部
    ctx.beginPath();
    ctx.moveTo(0, hy); ctx.lineTo(vx1, hy);
    ctx.moveTo(vx2, hy); ctx.lineTo(W, hy);
    ctx.stroke();

    // 中间虚线示意
    ctx.save();
    ctx.setLineDash([lw * 5, lw * 5]);
    ctx.strokeStyle = 'rgba(26,239,243,.35)';
    ctx.beginPath(); ctx.moveTo(vx1, hy); ctx.lineTo(vx2, hy); ctx.stroke();
    ctx.restore();

    // 标注
    const pad = lw * 3;
    const tag = (text, x, y, align) => {
      const w = ctx.measureText(text).width + pad * 2;
      const h = f + pad * 1.6;
      let bx = align === 'right' ? x - w : align === 'center' ? x - w / 2 : x;
      bx = clamp(bx, 0, W - w);
      const by = clamp(y - h / 2, 0, H - h);
      ctx.fillStyle = 'rgba(9,12,16,.82)';
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(bx, by, w, h, lw * 3);
      else ctx.rect(bx, by, w, h);
      ctx.fill();
      ctx.fillStyle = '#1AEFF3';
      ctx.fillText(text, bx + pad, by + h / 2);
    };
    tag(`眼睛高度 ${pct(GUIDE.hy)}`, W / 2, hy - f * 1.5, 'center');
    tag('25%', vx1, H - f * 1.2, 'center');
    tag('75%', vx2, H - f * 1.2, 'center');
  }

  if (opts.landmarks && face && face.count) {
    const px = (p) => [p.x * W, p.y * H];
    // 眼廓
    ctx.strokeStyle = 'rgba(255,92,92,.9)';
    ctx.lineWidth = lw;
    for (const set of [LM.eyeA, LM.eyeB]) {
      ctx.beginPath();
      set.forEach((i, k) => {
        const [x, y] = px(face.lm[i]);
        k ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      });
      ctx.closePath(); ctx.stroke();
    }
    // 双眼中点连线
    ctx.strokeStyle = 'rgba(255,92,92,.75)';
    ctx.beginPath();
    const [lx, ly] = px(face.eyeL), [rx, ry] = px(face.eyeR);
    ctx.moveTo(lx, ly); ctx.lineTo(rx, ry); ctx.stroke();
    // 瞳孔
    for (const e of [face.eyeL, face.eyeR]) {
      const [x, y] = px(e);
      ctx.fillStyle = 'rgba(255,92,92,.95)';
      ctx.beginPath(); ctx.arc(x, y, lw * 2.6, 0, Math.PI * 2); ctx.fill();
    }
    // 面部框
    ctx.strokeStyle = 'rgba(120,180,255,.75)';
    ctx.setLineDash([lw * 4, lw * 3]);
    ctx.lineWidth = lw;
    ctx.strokeRect(face.box.x0 * W, face.box.y0 * H,
                   (face.box.x1 - face.box.x0) * W, (face.box.y1 - face.box.y0) * H);
    ctx.setLineDash([]);
    // 中轴线
    ctx.strokeStyle = 'rgba(120,180,255,.4)';
    ctx.beginPath();
    ctx.moveTo(face.faceCx * W, 0); ctx.lineTo(face.faceCx * W, H); ctx.stroke();
  }
}

/* ============================================================
   渲染结果
   ============================================================ */
const ICON = { pass: '✓', fail: '✕', warn: '!', skip: '–' };

function render(checks) {
  const countable = checks.filter((c) => c.status !== 'skip');
  const passed = countable.filter((c) => c.status === 'pass').length;
  const failed = checks.filter((c) => c.status === 'fail');
  const warned = checks.filter((c) => c.status === 'warn');

  $('passCount').textContent = passed;
  $('totalCount').textContent = countable.length;

  const v = $('verdict');
  v.className = 'verdict';
  const title = $('verdictTitle'), sub = $('verdictSub');

  if (failed.length === 0 && warned.length === 0) {
    v.classList.add('ok');
    title.textContent = '全部自动检测项通过';
    sub.textContent = '仍请自行核对化妆、眼镜、遮挡、近期相貌等主观项。';
  } else if (failed.length === 0) {
    v.classList.add('warn');
    title.textContent = `${warned.length} 项需要注意`;
    sub.textContent = '没有硬性不合格项，但以下提示项建议一并处理。';
  } else {
    v.classList.add('bad');
    title.textContent = `${failed.length} 项不符合要求`;
    sub.textContent = '这些项很可能导致审核退回，建议先处理再上传。';
  }

  const fs = $('failSummary');
  if (failed.length) {
    fs.hidden = false;
    fs.textContent = '不符合：' + failed.map((c) => c.name).join('、');
  } else fs.hidden = true;

  // 分组渲染
  const order = ['文件与规格', '人像姿态', '背景与画质'];
  const groups = [...new Set([...order, ...checks.map((c) => c.group)])];
  const host = $('groups');
  host.innerHTML = '';
  for (const g of groups) {
    const items = checks.filter((c) => c.group === g);
    if (!items.length) continue;
    const sec = document.createElement('div');
    sec.className = 'group';
    const head = document.createElement('div');
    head.className = 'group-head';
    head.textContent = g;
    sec.appendChild(head);
    for (const it of items) {
      const row = document.createElement('div');
      row.className = `item ${it.status}`;
      row.innerHTML = `
        <div class="item-icon">${ICON[it.status] || ''}</div>
        <div class="item-body">
          <div class="item-name">${it.name}</div>
          <div class="item-req">${it.req}</div>
          ${it.hint ? `<div class="item-hint">${it.hint}</div>` : ''}
        </div>
        <div class="item-value">${it.value}</div>`;
      sec.appendChild(row);
    }
    host.appendChild(sec);
  }

  $('emptyState').hidden = true;
  $('report').hidden = false;
}

/* ============================================================
   主流程
   ============================================================ */
let current = null; // { file, img, px, face }

async function handleFile(file) {
  if (!file) return;
  if (!/^image\//.test(file.type) && !/\.(jpe?g)$/i.test(file.name)) {
    alert('请上传 JPG / JPEG 格式的照片。');
    return;
  }

  $('loading').hidden = false;
  $('loadingText').textContent = modelState === 'ready' ? '正在分析照片…' : '正在加载人脸模型（首次约 10 MB）…';

  try {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.decoding = 'sync';
    await new Promise((res, rej) => {
      img.onload = res; img.onerror = () => rej(new Error('图片解码失败'));
      img.src = url;
    });

    // 展示预览
    $('dropzone').hidden = true;
    $('previewWrap').hidden = false;
    $('previewImg').src = url;
    $('fileName').textContent = `${file.name} · ${img.naturalWidth}×${img.naturalHeight} · ${fmtSize(file.size)}`;

    const px = analyzePixels(img);

    await loadModel();
    $('loadingText').textContent = '正在分析照片…';
    const result = landmarker.detect(img);
    const face = analyzeFace(result, img);
    const checks = buildChecks(file, img, px, face);

    current = { file, img, px, face, checks };
    drawOverlay(img, face, {
      guides: $('toggleGuides').checked,
      landmarks: $('toggleLandmarks').checked,
    });
    render(checks);
  } catch (e) {
    console.error(e);
    alert('分析失败：' + (e && e.message ? e.message : e));
  } finally {
    $('loading').hidden = true;
  }
}

/* ============================================================
   事件绑定
   ============================================================ */
const dz = $('dropzone');
const fi = $('fileInput');

dz.addEventListener('click', () => fi.click());
dz.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fi.click(); }
});
fi.addEventListener('change', () => { handleFile(fi.files[0]); fi.value = ''; });

['dragenter', 'dragover'].forEach((ev) =>
  dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add('dragover'); }));
['dragleave', 'drop'].forEach((ev) =>
  dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove('dragover'); }));
dz.addEventListener('drop', (e) => {
  const f = e.dataTransfer.files && e.dataTransfer.files[0];
  if (f) handleFile(f);
});

$('resetBtn').addEventListener('click', () => {
  current = null;
  $('previewWrap').hidden = true;
  $('dropzone').hidden = false;
  $('report').hidden = true;
  $('emptyState').hidden = false;
  $('previewImg').removeAttribute('src');
});

$('toggleGuides').addEventListener('change', () => {
  if (current) drawOverlay(current.img, current.face, {
    guides: $('toggleGuides').checked,
    landmarks: $('toggleLandmarks').checked,
  });
});
$('toggleLandmarks').addEventListener('change', () => {
  if (current) drawOverlay(current.img, current.face, {
    guides: $('toggleGuides').checked,
    landmarks: $('toggleLandmarks').checked,
  });
});

/* 预热模型 */
if ('requestIdleCallback' in window) requestIdleCallback(() => loadModel().catch(() => {}));
else setTimeout(() => loadModel().catch(() => {}), 1200);

/* 供自动化测试调用 */
window.__checker = {
  analyze: async (dataUrl) => {
    const res = await fetch(dataUrl);
    const blob = await res.blob();
    const file = new File([blob], 'test.jpg', { type: 'image/jpeg' });
    await handleFile(file);
    return current;
  },
  state: () => ({ modelState, current }),
};
