// SignBridge Web Application (YOLOv11 + Turkish NLP + 3D-Style Avatar)

const YOLO_20 = [
  'ANNE','ARKADAS','BABA','DUR','EV','EVET','HAYIR','KARDES',
  'MERHABA','NASILSIN','NEREDE','OZUR DILEMEK','TAMAM','CEP TELEFONU',
  'TESEKKUR','TUVALET','YEMEK','ICMEK','IYI','KOTU'
];

const GLOSS_TO_TURKISH = {
  'MERHABA':'merhaba', 'TESEKKUR':'teşekkür ederim', 'EVET':'evet', 'HAYIR':'hayır',
  'IYI':'iyi', 'KOTU':'kötü', 'BUYUK':'büyük', 'KUCUK':'küçük',
  'BEN':'ben', 'SEN':'sen', 'BABA':'baba', 'ANNE':'anne', 'ABLA':'abla',
  'ABI':'abi', 'AMCA':'amca', 'KARDES':'kardeş', 'ARKADAS':'arkadaş', 'OGRETMEN':'öğretmen',
  'DOKTOR':'doktor', 'HEMSIRE':'hemşire', 'OKUL':'okul', 'EV':'ev', 'TUVALET':'tuvalet',
  'SU':'su', 'CAY':'çay', 'KAHVE':'kahve', 'EKMEK':'ekmek', 'YEMEK':'yemek',
  'CIKOLATA':'çikolata', 'BAKLAVA':'baklava', 'AYRAN':'ayran', 'ERIK':'erik',
  'SABAH':'sabah', 'AKSAM':'akşam', 'BUGUN':'bugün', 'DUN':'dün', 'GUN':'gün', 'KIS':'kış',
  'BIR':'bir', 'IKI':'iki', 'UC':'üç', 'DORT':'dört', 'BES':'beş',
  'ALTI':'altı', 'YEDI':'yedi', 'SEKIZ':'sekiz', 'DOKUZ':'dokuz', 'SIFIR':'sıfır',
  'OCAK':'ocak', 'SUBAT':'şubat', 'MART':'mart', 'NISAN':'nisan', 'MAYIS':'mayıs',
  'HAZIRAN':'haziran', 'TEMMUZ':'temmuz', 'AGUSTOS':'ağustos', 'EYLUL':'eylül',
  'EKIM':'ekim', 'KASIM':'kasım', 'ARALIK':'aralık',
  'PAZARTESI':'pazartesi', 'SALI':'salı', 'CARSAMBA':'çarşamba', 'PERSEMBE':'perşembe',
  'CUMA':'cuma', 'CUMARTESI':'cumartesi', 'PAZAR':'pazar',
  'ASK':'aşk', 'YARDIM':'yardım', 'DIKKAT':'dikkat', 'CEZA':'ceza',
  'FUTBOL':'futbol', 'VOLEYBOL':'voleybol', 'HENTBOL':'hentbol',
  'TV':'TV', 'CD':'CD', 'BILGISAYAR':'bilgisayar',
  'AKILLI':'akıllı', 'APTAL':'aptal', 'ZENGIN':'zengin',
  'ALERJI':'alerji', 'AYIP':'ayıp', 'ORUC':'oruç', 'FINAL':'final',
  'ALLAH':'Allah', 'AMIN':'amin', 'ELHAMDULILLAH':'elhamdülillah',
  'ATATURK':'Atatürk', 'ARAP':'Arap', 'BARIS':'barış',
  'EVLENMEK':'evlenmek', 'YATMAK':'yatmak', 'DOYMAK':'doymak',
  'HANGI':'hangi', 'CUNKU':'çünkü', 'YETER':'yeter',
  'HOSCA KAL':'hoşça kal', 'BOS VERMEK':'boş vermek', 'BU KADAR':'bu kadar',
  'CEP TELEFONU':'cep telefonu', 'LUTFEN':'lütfen', 'DUR':'dur',
  'NASILSIN':'nasılsın', 'NE':'ne', 'KAC':'kaç', 'NEREDE':'nerede',
  'YAPMAK':'yapmak', 'CALISMAK':'çalışmak', 'YORULMAK':'yorulmak',
  'BITMEK':'bitmek', 'ICMEK':'içmek', 'BULUSMAK':'buluşmak',
  'GORUSMEK':'görüşmek', 'BAKMAK':'bakmak',
  'COK':'çok', 'SAAT':'saat', 'SONRA':'sonra', 'BOS':'boş',
  'BERABER':'beraber', 'TAMAM':'tamam', 'PARK':'park', 'YAN':'yan', 'KAFE':'kafe', 'KENDI':'kendi'
};

const VERB_CONJ = {
  'YAPMAK':   { ben:'yapıyorum',   sen:'yapıyorsun',   _:'yapıyor' },
  'CALISMAK': { ben:'çalışıyorum', sen:'çalışıyorsun', _:'çalışıyor' },
  'YORULMAK': { ben:'yoruldum',    sen:'yoruldun',     _:'yoruldu' },
  'BITMEK':   { ben:'bitiyor',     sen:'bitiyor',      _:'bitiyor' },
  'ICMEK':    { ben:'içiyorum',    sen:'içiyorsun',    _:'içelim' },
  'BULUSMAK': { ben:'buluşalım',   sen:'buluşalım',    _:'buluşalım' },
  'GORUSMEK': { ben:'görüşürüz',   sen:'görüşürüz',    _:'görüşürüz' },
  'BAKMAK':   { ben:'bakıyorum',   sen:'bak',          _:'bak' },
};

const VOCAB_LIST = Object.keys(GLOSS_TO_TURKISH);

let sentence = ['MERHABA', 'BABA'];
let history = [
  { gloss: 'MERHABA NASILSIN', turk: 'Merhaba, nasılsın?' },
  { gloss: 'BEN IYI TESEKKUR', turk: 'Ben iyiyim, teşekkür ederim.' }
];
let currentPredWord = 'BABA';
let currentPredConf = 0.99;
let camRunning = false;
let camStream = null;
let animTick = 0;

// Avatar State
let avatarCurrentWord = 'BABA';
let avatarQueue = ['MERHABA', 'BABA'];
let avatarSpeed = 1.0;
let avatarPlaying = true;
let avatarPhase = 0;

function glossToSentence(glossWords) {
  if (!glossWords || glossWords.length === 0) return '—';
  const words = glossWords.filter((w, i) => i === 0 || w !== glossWords[i - 1]);
  const hasBen = words.includes('BEN');
  const hasSen = words.includes('SEN');
  const hasBeraber = words.includes('BERABER');
  const hasKendi = words.includes('KENDI');
  const out = [];
  let subj = hasBen ? 'ben' : (hasSen ? 'sen' : '_');

  for (let i = 0; i < words.length; i++) {
    const w = words[i];
    const next = words[i + 1];
    const next2 = words[i + 2];
    if (w === 'BES' && next === 'SONRA') {
      out.push('beşten sonra'); i++; continue;
    }
    if (w === 'SAAT' && next === 'KAC') {
      out.push('saat kaçta'); i++; continue;
    }
    if (w === 'PARK' && next === 'YAN' && next2 === 'KAFE') {
      out.push('parkın yanındaki kafede'); i += 2; continue;
    }
    if (w === 'KENDI') { out.push('kendine'); continue; }
    if (w === 'BOS' && !hasSen) { out.push('boşum'); continue; }
    if (w === 'IYI' && i === 0 && !hasKendi) { out.push('iyiyim'); continue; }
    if (VERB_CONJ[w]) {
      out.push(VERB_CONJ[w][subj] || VERB_CONJ[w]._);
      continue;
    }
    out.push(GLOSS_TO_TURKISH[w] || w.toLowerCase());
  }
  let res = out.join(' ');
  res = res.charAt(0).toUpperCase() + res.slice(1);
  if (hasBeraber && !res.endsWith('?')) res += ' mi?';
  else if (words.some(x => ['NE','KAC','NEREDE','NASILSIN'].includes(x))) res += '?';
  else res += '!';
  return res;
}

function switchTab(tab) {
  document.querySelectorAll('.view-section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('view-' + tab).classList.add('active');
  document.getElementById(tab === 'camera' ? 'tabBtnCamera' : 'tabBtnAvatar').classList.add('active');
}

function updateSentenceUI() {
  const box = document.getElementById('sentenceBox');
  const nlp = document.getElementById('nlpBox');
  if (sentence.length === 0) {
    box.className = 'empty';
    box.textContent = 'Henüz kelime algılanmadı...';
    nlp.textContent = '—';
  } else {
    box.className = '';
    box.innerHTML = sentence.map(w => `<span class="gloss-pill">${w}</span>`).join('');
    nlp.textContent = glossToSentence(sentence);
  }
  const hl = document.getElementById('historyList');
  hl.innerHTML = history.map(h => `
    <div class="history-item">
      <div style="color:#a78bfa;font-weight:700;">${h.turk}</div>
      <div style="font-size:0.72rem;color:#64748b;margin-top:2px;">${h.gloss}</div>
    </div>
  `).join('');
}

function addPhrase(words) {
  sentence = [...words];
  currentPredWord = words[words.length - 1];
  currentPredConf = 0.99;
  document.getElementById('predWord').textContent = currentPredWord;
  document.getElementById('predConf').textContent = '%99';
  highlightChip(currentPredWord);
  updateSentenceUI();
  if (document.getElementById('autoSpeak').checked) speakNow();
}

function manualAdd() {
  if (!currentPredWord) return;
  if (sentence[sentence.length - 1] !== currentPredWord) {
    sentence.push(currentPredWord);
    const bar = document.getElementById('cooldownBar');
    bar.style.width = '100%';
    setTimeout(() => { bar.style.width = '0%'; }, 900);
    updateSentenceUI();
  }
}

function clearSentence() {
  sentence = [];
  updateSentenceUI();
}

function saveSentence() {
  if (sentence.length === 0) return;
  const gloss = sentence.join(' ');
  const turk = glossToSentence(sentence);
  history.unshift({ gloss, turk });
  if (history.length > 10) history.pop();
  sentence = [];
  updateSentenceUI();
}

function speakNow() {
  const txt = glossToSentence(sentence);
  if (!txt || txt === '—' || !('speechSynthesis' in window)) return;
  window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(txt);
  u.lang = 'tr-TR';
  window.speechSynthesis.speak(u);
}

function highlightChip(word) {
  document.querySelectorAll('.word-chip').forEach(c => c.classList.remove('active'));
  const el = document.getElementById('chip_' + word);
  if (el) el.classList.add('active');
}

function filterWords() {
  const q = (document.getElementById('wordSearch').value || '').toUpperCase().trim();
  document.querySelectorAll('.word-chip').forEach(c => {
    const match = !q || c.dataset.gloss.includes(q) || c.textContent.toUpperCase().includes(q);
    c.style.display = match ? '' : 'none';
  });
}

function filterPoses() {
  const q = (document.getElementById('poseSearch').value || '').toUpperCase().trim();
  document.querySelectorAll('.pose-item').forEach(p => {
    const match = !q || p.dataset.gloss.includes(q);
    p.style.display = match ? '' : 'none';
  });
}

async function toggleCamera() {
  const btn = document.getElementById('btnStart');
  const ph = document.getElementById('camPlaceholder');
  const yoloCanvas = document.getElementById('yoloCanvas');
  const rawVideo = document.getElementById('rawVideo');

  if (camRunning) {
    camRunning = false;
    if (camStream) {
      camStream.getTracks().forEach(t => t.stop());
      camStream = null;
    }
    btn.textContent = 'Kamerayı Başlat';
    btn.classList.remove('active');
    return;
  }

  camRunning = true;
  btn.textContent = 'Kamerayı Durdur';
  btn.classList.add('active');
  ph.style.display = 'none';
  yoloCanvas.style.display = 'block';

  try {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      camStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
      rawVideo.srcObject = camStream;
      rawVideo.style.display = 'block';
      document.getElementById('rawFallbackCanvas').style.display = 'none';
    }
  } catch (err) {
    console.log('Camera fallback simulation enabled:', err);
  }
}

// Render Loop for Camera YOLO Overlay + 3D-Style Avatar
function renderLoop() {
  animTick += 0.04;
  drawCameraSimulation();
  drawAvatarScene();
  requestAnimationFrame(renderLoop);
}

function drawCameraSimulation() {
  const yoloCanvas = document.getElementById('yoloCanvas');
  const rawCanvas = document.getElementById('rawFallbackCanvas');
  const rawVideo = document.getElementById('rawVideo');

  if (rawCanvas && (!camStream)) {
    const rctx = rawCanvas.getContext('2d');
    const grad = rctx.createLinearGradient(0, 0, 640, 480);
    grad.addColorStop(0, '#16192b');
    grad.addColorStop(1, '#0d101d');
    rctx.fillStyle = grad;
    rctx.fillRect(0, 0, 640, 480);
    // Draw silhouette person
    rctx.fillStyle = '#262b44';
    rctx.beginPath();
    rctx.arc(320, 175, 58, 0, Math.PI * 2);
    rctx.fill();
    rctx.beginPath();
    rctx.roundRect(210, 245, 220, 210, 45);
    rctx.fill();
  }

  if (!yoloCanvas || yoloCanvas.style.display === 'none') return;
  const ctx = yoloCanvas.getContext('2d');
  if (camStream && rawVideo.readyState >= 2) {
    ctx.drawImage(rawVideo, 0, 0, 640, 480);
  } else {
    ctx.drawImage(rawCanvas, 0, 0, 640, 480);
  }

  // Draw YOLOv11 Person / Hand Bounding Box
  ctx.strokeStyle = '#22d3ee';
  ctx.lineWidth = 2.5;
  ctx.strokeRect(150, 85, 340, 360);
  ctx.fillStyle = 'rgba(34,211,238,0.85)';
  ctx.fillRect(150, 60, 195, 25);
  ctx.fillStyle = '#08090e';
  ctx.font = 'bold 13px Inter, sans-serif';
  ctx.fillText(`YOLOv11n: ${currentPredWord} (${Math.round(currentPredConf * 100)}%)`, 158, 77);

  // Draw Hand Landmarks Skeleton (21 points)
  const cx = 330 + Math.sin(animTick) * 18;
  const cy = 285 + Math.cos(animTick * 1.3) * 12;
  const pts = [
    [cx, cy + 55],
    [cx - 28, cy + 32], [cx - 45, cy + 10], [cx - 55, cy - 12], [cx - 62, cy - 30],
    [cx - 18, cy - 5],  [cx - 22, cy - 38], [cx - 24, cy - 62], [cx - 25, cy - 82],
    [cx - 2, cy - 8],   [cx - 2, cy - 44],  [cx - 2, cy - 70],  [cx - 2, cy - 90],
    [cx + 15, cy - 4],  [cx + 18, cy - 38], [cx + 20, cy - 62], [cx + 22, cy - 80],
    [cx + 30, cy + 6],  [cx + 38, cy - 22], [cx + 44, cy - 42], [cx + 48, cy - 58]
  ];
  const chains = [[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16],[0,17,18,19,20]];
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 2;
  chains.forEach(chain => {
    ctx.beginPath();
    chain.forEach((idx, i) => {
      if (i === 0) ctx.moveTo(pts[idx][0], pts[idx][1]);
      else ctx.lineTo(pts[idx][0], pts[idx][1]);
    });
    ctx.stroke();
  });
  pts.forEach(p => {
    ctx.fillStyle = '#ef4444';
    ctx.beginPath();
    ctx.arc(p[0], p[1], 4.2, 0, Math.PI * 2);
    ctx.fill();
  });
}

// Avatar Renderer
function drawAvatarScene() {
  const canvas = document.getElementById('avatarCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  // Stage Floor
  ctx.fillStyle = '#241b3b';
  ctx.beginPath();
  ctx.ellipse(w / 2, h - 40, 260, 55, 0, 0, Math.PI * 2);
  ctx.fill();

  if (avatarPlaying) {
    avatarPhase += 0.035 * avatarSpeed;
    const pct = Math.min(100, Math.round(((avatarPhase % (Math.PI * 2)) / (Math.PI * 2)) * 100));
    const pb = document.getElementById('avatarProgressBar');
    if (pb) pb.style.width = pct + '%';
  }

  // Word-dependent arm pose
  const hash = avatarCurrentWord.split('').reduce((a, c) => a + c.charCodeAt(0), 0);
  const swing = Math.sin(avatarPhase) * 0.45;
  const rightShoulder = { x: w / 2 - 48, y: 215 };
  const leftShoulder  = { x: w / 2 + 48, y: 215 };

  // Character Legs & Torso
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  // Legs
  ctx.strokeStyle = '#10243e';
  ctx.lineWidth = 26;
  ctx.beginPath(); ctx.moveTo(w / 2 - 24, 340); ctx.lineTo(w / 2 - 28, 475); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(w / 2 + 24, 340); ctx.lineTo(w / 2 + 28, 475); ctx.stroke();

  // Shoes
  ctx.fillStyle = '#9a3412';
  ctx.beginPath(); ctx.ellipse(w / 2 - 30, 485, 20, 10, 0, 0, Math.PI * 2); ctx.fill();
  ctx.beginPath(); ctx.ellipse(w / 2 + 30, 485, 20, 10, 0, 0, Math.PI * 2); ctx.fill();

  // Torso (White textured top + cyan scarf like SignBridge avatar)
  ctx.fillStyle = '#e2e8f0';
  ctx.beginPath();
  ctx.roundRect(w / 2 - 46, 195, 92, 150, 24);
  ctx.fill();

  // Cyan Scarf
  ctx.strokeStyle = '#0891b2';
  ctx.lineWidth = 12;
  ctx.beginPath();
  ctx.moveTo(w / 2 - 24, 198);
  ctx.quadraticCurveTo(w / 2, 215, w / 2 + 24, 198);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(w / 2 + 4, 206);
  ctx.lineTo(w / 2 + 10, 265);
  ctx.stroke();

  // Neck & Head
  ctx.fillStyle = '#5c3a21';
  ctx.fillRect(w / 2 - 12, 165, 24, 35);
  ctx.beginPath();
  ctx.arc(w / 2, 130, 42, 0, Math.PI * 2);
  ctx.fill();

  // Hair
  ctx.fillStyle = '#1c120c';
  ctx.beginPath();
  ctx.arc(w / 2, 116, 44, Math.PI * 0.95, Math.PI * 2.05);
  ctx.fill();

  // Eyes & Smile
  ctx.fillStyle = '#ffffff';
  ctx.beginPath(); ctx.arc(w / 2 - 15, 128, 8, 0, Math.PI * 2); ctx.fill();
  ctx.beginPath(); ctx.arc(w / 2 + 15, 128, 8, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = '#291709';
  ctx.beginPath(); ctx.arc(w / 2 - 15, 128, 4.5, 0, Math.PI * 2); ctx.fill();
  ctx.beginPath(); ctx.arc(w / 2 + 15, 128, 4.5, 0, Math.PI * 2); ctx.fill();

  // Right Signing Arm (Animated towards chin/forehead for TİD sign)
  const elbowX = rightShoulder.x - 42;
  const elbowY = rightShoulder.y - 10 + Math.cos(avatarPhase) * 14;
  const handX  = w / 2 - 10 + Math.sin(avatarPhase + (hash % 5)) * 22;
  const handY  = 160 + (hash % 3) * 18 + swing * 24;

  ctx.strokeStyle = '#5c3a21';
  ctx.lineWidth = 18;
  ctx.beginPath();
  ctx.moveTo(rightShoulder.x, rightShoulder.y);
  ctx.lineTo(elbowX, elbowY);
  ctx.lineTo(handX, handY);
  ctx.stroke();

  // Hand palm & fingers
  ctx.fillStyle = '#6e472b';
  ctx.beginPath();
  ctx.arc(handX, handY, 12, 0, Math.PI * 2);
  ctx.fill();

  // Left Arm
  ctx.strokeStyle = '#5c3a21';
  ctx.lineWidth = 18;
  ctx.beginPath();
  ctx.moveTo(leftShoulder.x, leftShoulder.y);
  ctx.lineTo(leftShoulder.x + 18, leftShoulder.y + 70);
  ctx.lineTo(leftShoulder.x + 14, leftShoulder.y + 135);
  ctx.stroke();
}

function playSingleAvatarWord(wordOverride) {
  const w = (wordOverride || document.getElementById('avatarWordInput').value || 'MERHABA').toUpperCase().trim();
  avatarCurrentWord = w;
  avatarPlaying = true;
  document.getElementById('avatarWordInput').value = w;
  document.getElementById('avatarWordDisplay').textContent = w;
  document.getElementById('avatarStatusBadge').textContent = 'Oynatılıyor: ' + w;
  document.getElementById('avatarAnimLabel').textContent = 'Anim: ' + w;
  document.querySelectorAll('.pose-item').forEach(p => {
    p.classList.toggle('active', p.dataset.gloss === w);
  });
}

function playAvatarSentence() {
  const raw = (document.getElementById('avatarSentenceInput').value || '').trim();
  if (!raw) return;
  const tokens = raw.toUpperCase().split(/\s+/);
  let i = 0;
  playSingleAvatarWord(tokens[0]);
  const timer = setInterval(() => {
    i++;
    if (i >= tokens.length) {
      clearInterval(timer);
      return;
    }
    playSingleAvatarWord(tokens[i]);
  }, Math.max(700, 1400 / avatarSpeed));
}

function playQuickAvatar(phrase) {
  document.getElementById('avatarSentenceInput').value = phrase;
  playAvatarSentence();
}

function stopAvatar() {
  avatarPlaying = false;
  document.getElementById('avatarStatusBadge').textContent = 'Durduruldu';
}

function updateAvatarSpeed(val) {
  avatarSpeed = parseFloat(val) || 1.0;
  document.getElementById('avatarSpeedVal').textContent = avatarSpeed.toFixed(1) + 'x';
}

function startAvatarMic() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {
    alert('Tarayıcınız ses tanımayı desteklemiyor.');
    return;
  }
  const rec = new SR();
  rec.lang = 'tr-TR';
  rec.onresult = (e) => {
    const text = e.results[0][0].transcript;
    document.getElementById('avatarSentenceInput').value = text;
    playAvatarSentence();
  };
  rec.start();
}

window.addEventListener('keydown', (e) => {
  if (e.code === 'Space' && document.activeElement.tagName !== 'INPUT') {
    e.preventDefault();
    manualAdd();
  }
});

window.addEventListener('DOMContentLoaded', () => {
  // Populate Camera word chips
  const wl = document.getElementById('wordList');
  document.getElementById('wordCountBadge').textContent = VOCAB_LIST.length;
  wl.innerHTML = VOCAB_LIST.map(w => {
    const tr = GLOSS_TO_TURKISH[w] || w.toLowerCase();
    const label = tr.charAt(0).toUpperCase() + tr.slice(1);
    return `<div class="word-chip ${w === 'BABA' ? 'active' : ''}" id="chip_${w}" data-gloss="${w}" onclick="addPhrase(['${w}'])">${label}</div>`;
  }).join('');

  // Populate Avatar pose sidebar
  const pl = document.getElementById('poseList');
  pl.innerHTML = VOCAB_LIST.map(w => `
    <div class="pose-item ${w === 'BABA' ? 'active' : ''}" data-gloss="${w}" onclick="playSingleAvatarWord('${w}')">
      <span>${w}</span>
      <span class="pose-tag">POZ</span>
    </div>
  `).join('');

  updateSentenceUI();
  yoloCanvas.style.display = 'block';
  document.getElementById('camPlaceholder').style.display = 'none';
  requestAnimationFrame(renderLoop);
});
