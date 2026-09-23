'use strict';

const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));

let STATE = null;
let REGIONS = {};        // {camera_id: {note, regions:[{x1,y1,x2,y2}]}}
let stream = null;       // EventSource của job đang chạy

// --- tiện ích ---------------------------------------------------------------

function toast(msg, kind = '') {
  const t = $('#toast');
  t.textContent = msg;
  t.className = 'toast ' + kind;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.add('hidden'), 5000);
}

function bytes(n) {
  if (n === null || n === undefined) return '';
  const u = ['B', 'KB', 'MB', 'GB'];
  let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(i ? 1 : 0)} ${u[i]}`;
}

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: opts.body && !(opts.body instanceof FormData)
      ? { 'Content-Type': 'application/json' } : undefined,
    ...opts,
  });
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = { detail: text }; }
  if (!res.ok) throw new Error((data && data.detail) || res.statusText);
  return data;
}

// --- điều hướng -------------------------------------------------------------

const SECTIONS = [
  ['sec-source', 'Nguồn'], ['sec-labels', 'Lớp nhãn'], ['sec-regions', 'Vùng bỏ qua'],
  ['sec-run', 'Chạy'], ['sec-cvat', 'CVAT'], ['sec-release', 'Phát hành'],
];

function navHeight() {
  return $('.steps').getBoundingClientRect().height;
}

function setActive(id) {
  $$('.steps a').forEach((a) => a.classList.toggle('active', a.getAttribute('href') === '#' + id));
}

function buildNav() {
  $('#stepnav').innerHTML = SECTIONS
    .map(([id, label]) => `<a href="#${id}">${label}</a>`).join('');

  // Khoá tạm việc tự dò khi người dùng vừa bấm, để cuộn mượt không nhảy
  // sang khu khác giữa chừng.
  let lockUntil = 0;

  $$('.steps a').forEach((a) => {
    a.onclick = (ev) => {
      ev.preventDefault();
      const id = a.getAttribute('href').slice(1);
      const top = $('#' + id).getBoundingClientRect().top + window.scrollY
                  - navHeight() - 14;
      lockUntil = Date.now() + 900;
      setActive(id);
      history.replaceState(null, '', '#' + id);
      window.scrollTo({ top: Math.max(top, 0), behavior: 'smooth' });
    };
  });

  // Khu đang xem = khu cuối cùng có đỉnh nằm trên mép dưới thanh điều hướng.
  // Cách này không phụ thuộc chiều cao khu, nên khu ngắn không bị bỏ qua.
  function update() {
    if (Date.now() < lockUntil) return;
    const line = navHeight() + 20;
    let cur = SECTIONS[0][0];
    for (const [id] of SECTIONS) {
      if ($('#' + id).getBoundingClientRect().top <= line) cur = id;
    }
    // chạm đáy trang thì luôn sáng khu cuối
    if (window.innerHeight + window.scrollY >= document.body.scrollHeight - 2) {
      cur = SECTIONS[SECTIONS.length - 1][0];
    }
    setActive(cur);
  }

  let ticking = false;
  const onScroll = () => {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(() => { ticking = false; update(); });
  };
  window.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('resize', onScroll);
  update();
}

// --- trạng thái -------------------------------------------------------------

async function refreshState() {
  STATE = await api('/api/state');

  $('#services').innerHTML = [
    ['MinIO', STATE.services.minio], ['CVAT', STATE.services.cvat],
  ].map(([n, up]) => `<span><i class="dot ${up ? 'up' : ''}"></i>${n}</span>`).join('');

  const r = STATE.released;
  $('#released').innerHTML = r
    ? `<div class="stat"><b>v${r.version}</b><span>${STATE.dataset_name}</span></div>
       <div class="stat"><b>${r.stats.num_images ?? '?'}</b><span>ảnh</span></div>
       <div class="stat"><b>${r.stats.num_boxes ?? '?'}</b><span>box</span></div>`
    : '<p class="hint">Chưa có bản phát hành nào trong <code>dataset/</code>.</p>';

  renderSources();
  renderRegionCameras();
  renderParams();
  renderPicker();
}

// Danh sách nguồn để chọn chạy. Số ảnh của lô hiện tại hiện ngay trên từng nguồn,
// thay cho dãy thẻ tổng cũ — thông tin theo nguồn hữu ích hơn con số gộp.
function renderPicker() {
  const box = $('#picksrc');
  if (!STATE.sources.length) {
    box.innerHTML = '<span class="none">Chưa khai nguồn nào ở khu 1.</span>';
    return;
  }
  const picked = new Set(pickedCameras());
  const per = STATE.batch.per_camera || {};
  box.innerHTML = STATE.sources.map((s) => {
    const n = per[s.camera_id];
    const on = picked.size === 0 || picked.has(s.camera_id);
    return `<label class="src ${on ? 'on' : ''}">
      <input type="checkbox" value="${s.camera_id}" ${on ? 'checked' : ''}>
      <strong>${s.camera_id}</strong>
      <span class="meta">${s.kind}${n ? ` · ${n} ảnh` : ''}${s.exists ? '' : ' · thiếu file'}</span>
    </label>`;
  }).join('') + `<button type="button" id="pickall">Chọn tất cả</button>`;

  box.querySelectorAll('input').forEach((cb) => {
    cb.onchange = () => {
      cb.closest('.src').classList.toggle('on', cb.checked);
      if (!box.querySelector('input:checked')) {
        toast('Phải chọn ít nhất một nguồn', 'bad');
        cb.checked = true;
        cb.closest('.src').classList.add('on');
      }
    };
  });
  $('#pickall').onclick = () => {
    box.querySelectorAll('input').forEach((cb) => {
      cb.checked = true;
      cb.closest('.src').classList.add('on');
    });
  };
}

function pickedCameras() {
  return [...document.querySelectorAll('#picksrc input:checked')].map((cb) => cb.value);
}

function renderParams() {
  const f = $('#params');
  if (f.dataset.dirty === '1') return;      // đang sửa dở thì đừng ghi đè
  for (const [k, v] of Object.entries(STATE.sampling || {})) {
    if (f.elements[k]) f.elements[k].value = v;
  }
}

function renderSources() {
  const tb = $('#srctable tbody');
  if (!STATE.sources.length) {
    tb.innerHTML = '<tr><td class="hint">Chưa khai nguồn nào. Thêm ở khung bên phải.</td></tr>';
    return;
  }
  tb.innerHTML = STATE.sources.map((s) => {
    const scene = Object.entries(s.scene || {})
      .map(([k, v]) => `${k}=${v}`).join(', ');
    const rbadge = s.regions_stale
      ? `<span class="badge bad" title="vẽ cho ${s.regions_source}">${s.regions} vùng — của nguồn khác</span>`
      : `<span class="badge ${s.regions ? 'ok' : ''}">${s.regions} vùng</span>`;
    return `<tr>
      <td><strong>${s.camera_id}</strong></td>
      <td><span class="badge ${s.kind === 'video' ? 'video' : ''}">${s.kind}</span></td>
      <td><code>${s.path}</code>${s.exists ? '' : ' <span class="badge bad">thiếu file</span>'}</td>
      <td class="hint">${scene}</td>
      <td>${rbadge}</td>
      <td><button data-del="${s.camera_id}">Xoá</button></td>
    </tr>`;
  }).join('');

  renderOrphans();
  tb.querySelectorAll('[data-del]').forEach((btn) => {
    btn.onclick = async () => {
      const cam = btn.dataset.del;
      const n = (STATE.sources.find((s) => s.camera_id === cam) || {}).regions || 0;
      const extra = n ? `\n\n${n} vùng bỏ qua của ${cam} cũng bị xoá theo.` : '';
      if (!confirm(`Xoá nguồn ${cam} khỏi pipeline.yaml?${extra}`)) return;
      try {
        const d = await api('/api/sources/' + cam, { method: 'DELETE' });
        toast(d.regions_dropped
          ? `Đã xoá nguồn ${cam} và ${d.regions_dropped} vùng bỏ qua` : `Đã xoá nguồn ${cam}`, 'ok');
        await loadRegions();
        await refreshState();
      } catch (e) { toast(e.message, 'bad'); }
    };
  });
}

async function renderOrphans() {
  const box = $('#orphans');
  let d;
  try { d = await api('/api/regions/orphans'); } catch { box.classList.add('hidden'); return; }
  if (!d.orphans.length) { box.classList.add('hidden'); return; }
  box.classList.remove('hidden');
  box.innerHTML = `<strong>Vùng bỏ qua mồ côi:</strong> ${d.orphans.join(', ')} —
    còn vùng nhưng không còn nguồn nào khai báo. Nếu sau này dùng lại những
    <code>camera_id</code> đó cho dữ liệu khác, chúng sẽ thừa kế vùng cũ.
    <button id="dropall">Xoá hết</button>`;
  $('#dropall').onclick = async () => {
    if (!confirm(`Xoá vùng bỏ qua của: ${d.orphans.join(', ')}?`)) return;
    for (const cam of d.orphans) {
      try { await api('/api/regions/' + cam, { method: 'DELETE' }); } catch (e) { toast(e.message, 'bad'); }
    }
    toast('Đã dọn vùng mồ côi', 'ok');
    await loadRegions();
    await refreshState();
  };
}

// --- inbox ------------------------------------------------------------------

async function refreshInbox() {
  const d = await api('/api/inbox');
  $('#inboxdir').textContent = 'Lưu vào ' + d.dir + '/';
  const ul = $('#inbox');
  if (!d.entries.length) {
    ul.innerHTML = '<li class="hint">Trống</li>';
    return;
  }
  ul.innerHTML = d.entries.map((e) => `<li>
      <span class="badge ${e.is_video ? 'video' : ''}">${e.is_video ? 'video' : (e.images !== undefined ? e.images + ' ảnh' : 'file')}</span>
      <span class="name">${e.name}</span>
      <span class="size">${bytes(e.size)}</span>
      <button data-use="${e.name}" data-video="${e.is_video}">Dùng</button>
      <button data-rm="${e.name}">✕</button>
    </li>`).join('');

  ul.querySelectorAll('[data-use]').forEach((b) => {
    b.onclick = () => {
      $('#srcpath').value = d.dir + '/' + b.dataset.use;
      $('#srcform').kind.value = b.dataset.video === 'true' ? 'video' : 'images';
      if (!$('#srcform').camera_id.value) {
        const used = new Set(STATE.sources.map((s) => s.camera_id));
        let i = 1;
        while (used.has('cam' + String(i).padStart(2, '0'))) i++;
        $('#srcform').camera_id.value = 'cam' + String(i).padStart(2, '0');
      }
      $('#srcform').camera_id.focus();
      toast('Đã điền đường dẫn — đặt camera_id rồi bấm Lưu nguồn');
    };
  });
  ul.querySelectorAll('[data-rm]').forEach((b) => {
    b.onclick = async () => {
      if (!confirm('Xoá ' + b.dataset.rm + '?')) return;
      try {
        await api('/api/inbox/' + encodeURIComponent(b.dataset.rm), { method: 'DELETE' });
        toast('Đã xoá', 'ok');
        refreshInbox();
      } catch (e) { toast(e.message, 'bad'); }
    };
  });
}

function setupUpload() {
  const drop = $('#drop');
  const input = $('#file');
  $('#pick').onclick = () => input.click();
  input.onchange = () => input.files[0] && upload(input.files[0]);

  ['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => {
    e.preventDefault(); drop.classList.add('over');
  }));
  ['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => {
    e.preventDefault(); drop.classList.remove('over');
  }));
  drop.addEventListener('drop', (e) => {
    const f = e.dataTransfer.files[0];
    if (f) upload(f);
  });
}

function upload(file) {
  const bar = $('#upprog');
  bar.classList.remove('hidden');
  const fill = bar.querySelector('div');
  const txt = bar.querySelector('span');

  const fd = new FormData();
  fd.append('file', file);
  const xhr = new XMLHttpRequest();
  xhr.open('POST', '/api/inbox/upload');
  xhr.upload.onprogress = (e) => {
    if (!e.lengthComputable) return;
    const p = (e.loaded / e.total) * 100;
    fill.style.width = p + '%';
    txt.textContent = `${file.name} — ${p.toFixed(0)}%`;
  };
  xhr.onload = () => {
    bar.classList.add('hidden');
    fill.style.width = '0';
    if (xhr.status < 400) {
      toast('Đã tải lên ' + JSON.parse(xhr.responseText).name, 'ok');
      refreshInbox();
    } else {
      let d; try { d = JSON.parse(xhr.responseText).detail; } catch { d = xhr.statusText; }
      toast(d, 'bad');
    }
  };
  xhr.onerror = () => { bar.classList.add('hidden'); toast('Tải lên thất bại', 'bad'); };
  xhr.send(fd);
}

function setupSourceForm() {
  $('#srcform').onsubmit = async (e) => {
    e.preventDefault();
    const f = e.target;
    try {
      await api('/api/sources', {
        method: 'POST',
        body: JSON.stringify({
          camera_id: f.camera_id.value.trim(),
          kind: f.kind.value,
          path: f.path.value.trim(),
          scene: {
            weather: f.weather.value.trim(),
            camera_state: f.camera_state.value.trim(),
            location: f.location.value.trim(),
          },
        }),
      });
      toast('Đã ghi vào configs/pipeline.yaml', 'ok');
      f.reset();
      await refreshState();
    } catch (err) { toast(err.message, 'bad'); }
  };
}

// --- lớp nhãn ----------------------------------------------------------------

let LABELS = { labels: [], class_map: [], model_classes: {}, model_known: false };
let LABELS_SAVED = '';

function labelNames() {
  return [...document.querySelectorAll('#labellist input')].map((i) => i.value.trim());
}

function mapRows() {
  return [...document.querySelectorAll('#maplist tr')].map((tr) => ({
    coco_id: Number(tr.dataset.coco),
    label: tr.querySelector('select').value,
  }));
}

async function loadLabels() {
  LABELS = await api('/api/labels');
  LABELS_SAVED = JSON.stringify(LABELS.labels.map((l) => l.name));
  renderLabels();
}

function renderLabels() {
  const names = LABELS.labels.map((l) => l.name);

  $('#labellist').innerHTML = names.map((n, i) => `<li>
      <span class="idx" title="class_id trong file nhãn YOLO">${i}</span>
      <input value="${n}">
      <button data-up="${i}" ${i === 0 ? 'disabled' : ''} title="lên">↑</button>
      <button data-down="${i}" ${i === names.length - 1 ? 'disabled' : ''} title="xuống">↓</button>
      <button data-rm="${i}" ${names.length === 1 ? 'disabled' : ''}>✕</button>
    </li>`).join('');

  const opts = (sel) => names.map((n) =>
    `<option value="${n}" ${n === sel ? 'selected' : ''}>${n}</option>`).join('');

  $('#maplist').innerHTML = LABELS.class_map.map((m) => `<tr data-coco="${m.coco_id}">
      <td><strong>${m.coco_id}</strong></td>
      <td class="hint">${m.coco_name || '—'}${
        LABELS.model_known && !m.in_model
          ? ' <span class="badge bad">model không có id này</span>' : ''}</td>
      <td>→</td>
      <td><select>${opts(m.label)}</select></td>
      <td><button data-rmmap="${m.coco_id}">✕</button></td>
    </tr>`).join('') || '<tr><td class="hint">Chưa có ánh xạ nào — bước gán nhãn sơ bộ sẽ không sinh box nào.</td></tr>';

  // danh sách lớp model, bỏ những lớp đã ánh xạ rồi
  const used = new Set(LABELS.class_map.map((m) => String(m.coco_id)));
  const free = Object.entries(LABELS.model_classes || {}).filter(([id]) => !used.has(id));
  $('#newcoco').innerHTML = free.length
    ? free.map(([id, n]) => `<option value="${id}">${id} — ${n}</option>`).join('')
    : '<option value="">(đã ánh xạ hết)</option>';
  $('#addmap').disabled = !free.length;

  $('#modelinfo').innerHTML = LABELS.model_known
    ? `Model phân biệt <strong>${Object.keys(LABELS.model_classes).length} lớp</strong>.
       Lớp nào không khai ở đây thì bị bỏ khi gán nhãn sơ bộ.`
    : `<span class="badge bad">chưa đọc được lớp của model</span>
       Danh sách dưới là bảng COCO rút gọn — kiểm lại <code>model</code> trong
       <code>configs/prelabel.yaml</code>.`;

  $('#newcocolabel').innerHTML = opts(names[0]);

  // Hai kiểu thay đổi, hậu quả khác hẳn nhau:
  //  - đổi thứ tự  -> file nhãn sơ bộ (.txt chứa SỐ) trỏ sai; CVAT không sao vì
  //                   bước phát hành tra theo TÊN lớp rồi đánh số lại
  //  - đổi tên/xoá -> nhãn trên CVAT còn mang tên cũ, bước phát hành sẽ chặn cả lô
  const warn = $('#labelwarn');
  const saved = JSON.parse(LABELS_SAVED || '[]');
  const gone = saved.filter((n) => !names.includes(n));
  const reordered = gone.length === 0 && JSON.stringify(names) !== LABELS_SAVED;

  warn.classList.toggle('hidden', !gone.length && !reordered);
  if (gone.length) {
    warn.innerHTML = `<strong>Đã bỏ hoặc đổi tên lớp: ${gone.map((n) => `<code>${n}</code>`).join(', ')}</strong>.
      Nhãn nào trên CVAT còn mang tên cũ sẽ làm bước phát hành <strong>chặn cả lô</strong>
      (“lớp ... không có trong label spec”). Đổi chúng sang lớp khác trong CVAT trước.
      Nhãn sơ bộ cũng cần chạy lại.`;
  } else if (reordered) {
    warn.innerHTML = `<strong>Đổi thứ tự lớp</strong> làm <code>class_id</code> đổi theo, nên
      file nhãn sơ bộ đã sinh sẽ trỏ sai lớp — chạy lại “Gán nhãn sơ bộ”.
      Nhãn người đã sửa trên CVAT <em>không</em> bị ảnh hưởng: bước phát hành tra theo tên
      lớp rồi đánh số lại.`;
  }

  $('#labellist').querySelectorAll('input').forEach((inp, i) => {
    inp.oninput = () => { LABELS.labels[i].name = inp.value; };
    inp.onblur = () => { LABELS.labels[i].name = inp.value.trim(); renderLabels(); };
  });
  const move = (i, j) => {
    const a = LABELS.labels;
    [a[i], a[j]] = [a[j], a[i]];
    renderLabels();
  };
  $('#labellist').querySelectorAll('[data-up]').forEach((b) =>
    b.onclick = () => move(+b.dataset.up, +b.dataset.up - 1));
  $('#labellist').querySelectorAll('[data-down]').forEach((b) =>
    b.onclick = () => move(+b.dataset.down, +b.dataset.down + 1));
  $('#labellist').querySelectorAll('[data-rm]').forEach((b) =>
    b.onclick = () => {
      const gone = LABELS.labels[+b.dataset.rm].name;
      LABELS.labels.splice(+b.dataset.rm, 1);
      // ánh xạ trỏ tới lớp vừa xoá thì bỏ luôn, nếu không sẽ không lưu được
      LABELS.class_map = LABELS.class_map.filter((m) => m.label !== gone);
      renderLabels();
    });
  $('#maplist').querySelectorAll('select').forEach((sel, i) =>
    sel.onchange = () => { LABELS.class_map[i].label = sel.value; });
  $('#maplist').querySelectorAll('[data-rmmap]').forEach((b) =>
    b.onclick = () => {
      LABELS.class_map = LABELS.class_map.filter((m) => m.coco_id !== +b.dataset.rmmap);
      renderLabels();
    });
}

function setupLabels() {
  $('#addlabel').onclick = () => {
    const n = $('#newlabel').value.trim();
    if (!n) return toast('Nhập tên lớp đã', 'bad');
    if (LABELS.labels.some((l) => l.name === n)) return toast('Lớp này đã có', 'bad');
    LABELS.labels.push({ name: n, type: 'rectangle' });
    $('#newlabel').value = '';
    renderLabels();
  };
  $('#newlabel').onkeydown = (e) => { if (e.key === 'Enter') { e.preventDefault(); $('#addlabel').click(); } };

  $('#addmap').onclick = () => {
    const raw = $('#newcoco').value;
    if (raw === '') return;
    const id = Number(raw);
    if (LABELS.class_map.some((m) => m.coco_id === id)) return toast(`lớp ${id} đã có ánh xạ`, 'bad');
    LABELS.class_map.push({ coco_id: id, coco_name: LABELS.model_classes[raw] || null,
                            label: $('#newcocolabel').value, in_model: true });
    LABELS.class_map.sort((a, b) => a.coco_id - b.coco_id);
    renderLabels();
  };

  $('#resetlabels').onclick = async () => { await loadLabels(); toast('Đã khôi phục bản đang lưu'); };

  $('#savelabels').onclick = async () => {
    try {
      const d = await api('/api/labels', {
        method: 'PUT',
        body: JSON.stringify({ labels: labelNames(), class_map: mapRows() }),
      });
      LABELS = d;
      LABELS_SAVED = JSON.stringify(LABELS.labels.map((l) => l.name));
      renderLabels();
      toast(`Đã lưu ${d.labels.length} lớp và ${d.class_map.length} ánh xạ`, 'ok');
    } catch (e) { toast(e.message, 'bad'); }
  };
}

// --- khoanh vùng ------------------------------------------------------------

const RE = {
  cam: null, img: null, regions: [], sel: -1,
  drag: null, undo: [],
};

function renderRegionCameras() {
  const sel = $('#rcam');
  const prev = sel.value;
  sel.innerHTML = '<option value="">— chọn —</option>' + STATE.sources
    .map((s) => `<option value="${s.camera_id}">${s.camera_id} (${s.kind})${s.exists ? '' : ' — thiếu file'}</option>`)
    .join('');
  if (prev && STATE.sources.some((s) => s.camera_id === prev)) sel.value = prev;
}

async function loadRegions() {
  const d = await api('/api/regions');
  REGIONS = d.cameras || {};
}

function loadPreview() {
  const cam = $('#rcam').value;
  RE.cam = cam;
  RE.sel = -1;
  RE.undo = [];
  const msg = $('#canvasmsg');
  if (!cam) {
    RE.img = null;
    msg.textContent = 'Chọn một camera để bắt đầu';
    msg.classList.remove('hidden');
    draw();
    return;
  }
  RE.regions = JSON.parse(JSON.stringify(REGIONS[cam]?.regions || []));

  const src = STATE.sources.find((s) => s.camera_id === cam);
  const warn = $('#rwarn');
  if (src && src.regions_stale) {
    warn.classList.remove('hidden');
    warn.innerHTML = `<strong>${RE.regions.length} vùng này vẽ cho nguồn khác</strong>
      (<code>${src.regions_source}</code>), nguồn hiện tại là <code>${src.path}</code>.
      Chúng đang được áp lên dữ liệu mới. Vẽ lại hoặc xoá đi.
      <button id="rdropstale">Xoá hết vùng cũ</button>`;
    $('#rdropstale').onclick = () => {
      RE.undo.push(JSON.parse(JSON.stringify(RE.regions)));
      RE.regions = [];
      RE.sel = -1;
      draw();
      toast('Đã xoá trên màn hình — bấm “Lưu vùng” để ghi lại');
    };
  } else {
    warn.classList.add('hidden');
  }
  const pos = Number($('#rpos').value) / 100;
  $('#rposlabel').textContent = `${(pos * 100).toFixed(0)}% thời lượng`;
  msg.textContent = 'Đang tải khung hình…';
  msg.classList.remove('hidden');

  const img = new Image();
  img.onload = () => {
    RE.img = img;
    msg.classList.add('hidden');
    draw();
  };
  img.onerror = () => {
    RE.img = null;
    msg.textContent = 'Không lấy được khung hình của camera này';
    msg.classList.remove('hidden');
    draw();
  };
  img.src = `/api/preview/${encodeURIComponent(cam)}?pos=${pos}&_=${Date.now()}`;
}

function draw() {
  const c = $('#canvas');
  const ctx = c.getContext('2d');
  if (!RE.img) { c.width = 0; c.height = 0; return; }

  const maxW = $('#canvaswrap').clientWidth;
  const scale = Math.min(1, maxW / RE.img.naturalWidth);
  c.width = Math.round(RE.img.naturalWidth * scale);
  c.height = Math.round(RE.img.naturalHeight * scale);
  ctx.drawImage(RE.img, 0, 0, c.width, c.height);

  RE.regions.forEach((r, i) => {
    const x = r.x1 * c.width, y = r.y1 * c.height;
    const w = (r.x2 - r.x1) * c.width, h = (r.y2 - r.y1) * c.height;
    ctx.fillStyle = i === RE.sel ? 'rgba(255,190,60,.34)' : 'rgba(255,60,60,.28)';
    ctx.fillRect(x, y, w, h);
    ctx.lineWidth = i === RE.sel ? 3 : 2;
    ctx.strokeStyle = i === RE.sel ? '#ffbe3c' : '#ff4646';
    ctx.strokeRect(x, y, w, h);
    ctx.fillStyle = '#fff';
    ctx.font = '600 12px ui-sans-serif, system-ui, sans-serif';
    ctx.fillText(String(i + 1), x + 5, y + 15);
  });

  if (RE.drag) {
    const d = RE.drag;
    ctx.setLineDash([6, 4]);
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#ffbe3c';
    ctx.strokeRect(d.x * c.width, d.y * c.height,
                   (d.x2 - d.x) * c.width, (d.y2 - d.y) * c.height);
    ctx.setLineDash([]);
  }
}

function canvasPos(ev) {
  const c = $('#canvas');
  const r = c.getBoundingClientRect();
  return {
    x: Math.min(Math.max((ev.clientX - r.left) / r.width, 0), 1),
    y: Math.min(Math.max((ev.clientY - r.top) / r.height, 0), 1),
  };
}

function hitTest(p) {
  for (let i = RE.regions.length - 1; i >= 0; i--) {
    const r = RE.regions[i];
    if (p.x >= r.x1 && p.x <= r.x2 && p.y >= r.y1 && p.y <= r.y2) return i;
  }
  return -1;
}

function setupCanvas() {
  const c = $('#canvas');

  c.addEventListener('pointerdown', (ev) => {
    if (!RE.img) return;
    const p = canvasPos(ev);
    const hit = hitTest(p);
    if (hit >= 0 && !ev.shiftKey) {     // bấm vào vùng có sẵn = chọn
      RE.sel = hit;
      draw();
      return;
    }
    c.setPointerCapture(ev.pointerId);
    RE.drag = { x: p.x, y: p.y, x2: p.x, y2: p.y };
    RE.sel = -1;
  });

  // Theo dõi trên window chứ không phải canvas: thả chuột ngoài khung hình
  // (hoặc chuyển tab giữa chừng) vẫn kết thúc được vùng đang vẽ.
  window.addEventListener('pointermove', (ev) => {
    if (!RE.drag) return;
    const p = canvasPos(ev);
    RE.drag.x2 = p.x;
    RE.drag.y2 = p.y;
    draw();
  });

  function endDrag() {
    if (!RE.drag) return;
    const d = RE.drag;
    RE.drag = null;
    const r = {
      x1: Math.min(d.x, d.x2), y1: Math.min(d.y, d.y2),
      x2: Math.max(d.x, d.x2), y2: Math.max(d.y, d.y2),
    };
    if (r.x2 - r.x1 > 0.005 && r.y2 - r.y1 > 0.005) {
      RE.undo.push(JSON.parse(JSON.stringify(RE.regions)));
      RE.regions.push(r);
      RE.sel = RE.regions.length - 1;
    }
    draw();
  }

  window.addEventListener('pointerup', endDrag);
  window.addEventListener('pointercancel', endDrag);
  window.addEventListener('blur', endDrag);

  window.addEventListener('keydown', (ev) => {
    if (ev.target.matches('input, select, textarea')) return;
    if ((ev.key === 'Delete' || ev.key === 'Backspace') && RE.sel >= 0) {
      ev.preventDefault();
      RE.undo.push(JSON.parse(JSON.stringify(RE.regions)));
      RE.regions.splice(RE.sel, 1);
      RE.sel = -1;
      draw();
    }
  });

  window.addEventListener('resize', draw);

  $('#rcam').onchange = loadPreview;
  $('#rpos').oninput = () => {
    $('#rposlabel').textContent = `${$('#rpos').value}% thời lượng`;
  };
  $('#rpos').onchange = () => { if (RE.cam) loadPreview(); };

  $('#rundo').onclick = () => {
    if (!RE.undo.length) return toast('Không còn gì để hoàn tác');
    RE.regions = RE.undo.pop();
    RE.sel = -1;
    draw();
  };
  $('#rclear').onclick = () => {
    if (!RE.regions.length) return;
    RE.undo.push(JSON.parse(JSON.stringify(RE.regions)));
    RE.regions = [];
    RE.sel = -1;
    draw();
  };
  $('#rsave').onclick = async () => {
    if (!RE.cam) return toast('Chọn camera đã', 'bad');
    try {
      const d = await api('/api/regions/' + encodeURIComponent(RE.cam), {
        method: 'PUT',
        body: JSON.stringify({ regions: RE.regions }),
      });
      toast(`Đã lưu ${d.count} vùng cho ${RE.cam}`, 'ok');
      await loadRegions();
      renderSources();
    } catch (e) { toast(e.message, 'bad'); }
  };
}

// --- chạy bước --------------------------------------------------------------

function logLine(text, cls = '') {
  const log = $('#log');
  const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 40;
  const span = document.createElement('span');
  span.className = cls;
  span.textContent = text + '\n';
  log.appendChild(span);
  if (atBottom) log.scrollTop = log.scrollHeight;
}

function classify(line) {
  if (line.startsWith('$ ')) return 'cmd';
  if (/LỖI|Traceback|Error|error:/.test(line)) return 'err';
  if (/CẢNH BÁO|WARNING/.test(line)) return 'warn';
  return '';
}

function setRunning(on) {
  $$('.step').forEach((b) => { b.disabled = on; });
  $('#runall').disabled = on;
  $('#cancel').classList.toggle('hidden', !on);
}

async function runStep(step, extra = {}) {
  const body = { ...extra };
  if (step === 'ingest' || step === 'cvat') {
    const cams = pickedCameras();
    // gửi rỗng khi chọn hết = để script tự chạy mọi nguồn
    if (cams.length && cams.length < STATE.sources.length) body.cameras = cams;
  }
  if (step === 'ingest') body.upload = $('#doupload').checked;
  if (step === 'release' || step === 'publish') {
    body.version = $('#version').value.trim();
    const t = $('#reltask').value.trim();
    if (t && step === 'release') body.task = Number(t);
  }
  let job;
  try {
    job = await api('/api/run/' + step, { method: 'POST', body: JSON.stringify(body) });
  } catch (e) {
    toast(e.message, 'bad');
    throw e;
  }
  setRunning(true);
  $('#jobtitle').textContent = `${job.label} — đang chạy`;
  logLine(`\n───── ${job.label} ─────`);

  return new Promise((resolve, reject) => {
    if (stream) stream.close();
    stream = new EventSource(`/api/job/${job.id}/stream`);
    stream.onmessage = (e) => {
      const line = JSON.parse(e.data);
      logLine(line, classify(line));
    };
    stream.addEventListener('done', (e) => {
      const j = JSON.parse(e.data);
      stream.close();
      stream = null;
      setRunning(false);
      const ok = j.state === 'ok';
      $('#jobtitle').textContent = `${j.label} — ${ok ? 'xong' : j.state} (${j.elapsed}s)`;
      logLine(ok ? `✓ xong sau ${j.elapsed}s` : `✗ ${j.state} (mã ${j.returncode})`,
              ok ? 'ok' : 'err');
      toast(ok ? `${j.label}: xong` : `${j.label}: thất bại — xem log`, ok ? 'ok' : 'bad');
      refreshState();
      refreshCvat();
      ok ? resolve(j) : reject(new Error(j.state));
    });
    stream.onerror = () => {
      if (stream) { stream.close(); stream = null; }
      setRunning(false);
      reject(new Error('mất kết nối log'));
    };
  });
}

function setupParams() {
  const f = $('#params');
  f.addEventListener('input', () => { f.dataset.dirty = '1'; });
  $('#paramsreset').onclick = () => {
    f.dataset.dirty = '0';
    renderParams();
    toast('Đã khôi phục giá trị đang lưu');
  };
  f.onsubmit = async (e) => {
    e.preventDefault();
    try {
      const d = await api('/api/sampling', {
        method: 'PATCH',
        body: JSON.stringify({
          sample_every_n: Number(f.sample_every_n.value),
          max_per_camera: Number(f.max_per_camera.value),
          source_fps: Number(f.source_fps.value),
          image_glob: f.image_glob.value.trim(),
        }),
      });
      f.dataset.dirty = '0';
      toast(`Đã lưu: 1 frame mỗi ${d.sampling.sample_every_n}, tối đa ${d.sampling.max_per_camera || '∞'}/camera`, 'ok');
      await refreshState();
    } catch (err) { toast(err.message, 'bad'); }
  };
}

function setupRun() {
  $$('.step').forEach((b) => {
    b.onclick = () => runStep(b.dataset.step).catch(() => {});
  });
  $('#runall').onclick = async () => {
    for (const s of ['ingest', 'prelabel', 'package', 'cvat']) {
      try { await runStep(s); } catch { toast('Dừng vì bước ' + s + ' thất bại', 'bad'); return; }
    }
    toast('Xong 4 bước — sang CVAT để soát nhãn', 'ok');
  };
  $('#cancel').onclick = async () => {
    try { await api('/api/job/cancel', { method: 'POST' }); } catch (e) { toast(e.message, 'bad'); }
  };
  $('#clearlog').onclick = () => { $('#log').textContent = ''; };
}

// --- CVAT -------------------------------------------------------------------

async function refreshCvat() {
  const el = $('#cvat');
  let d;
  try { d = await api('/api/cvat'); } catch (e) { el.innerHTML = `<p class="hint">${e.message}</p>`; return; }
  if (!d.ok) {
    el.innerHTML = `<p class="hint">Không kết nối được CVAT — ${d.error}</p>`;
    return;
  }
  if (!d.tasks.length) {
    el.innerHTML = '<p class="hint">Chưa có task nào. Chạy bước “Tạo task CVAT” ở trên.</p>';
    return;
  }
  el.innerHTML = d.tasks.map((t) => `
    <div class="task">
      <div class="taskhead">
        <strong>#${t.id}</strong>
        <span>${t.name}</span>
        <span class="badge">${t.size} ảnh</span>
        <span class="badge ${t.passed ? 'ok' : ''}">${t.jobs.filter((j) => j.passed).length}/${t.jobs.length} PASS</span>
        <a href="${t.url}" target="_blank" rel="noopener">mở trong CVAT ↗</a>
        <span style="flex:1"></span>
        <button class="danger" data-rmtask="${t.id}" data-name="${t.name}">Xoá task</button>
      </div>
      ${t.jobs.map((j) => `<div class="job">
          <span>job ${j.id}</span>
          <span class="st">${j.stage}/${j.state} · frame ${j.frames[0]}–${j.frames[1]}</span>
          <span class="badge ${j.passed ? 'ok' : ''}">${j.passed ? 'PASS' : 'chưa'}</span>
          <span style="flex:1"></span>
          <button data-job="${j.id}" data-dec="pass">PASS</button>
          <button data-job="${j.id}" data-dec="fail">FAIL</button>
        </div>`).join('')}
    </div>`).join('');

  el.querySelectorAll('[data-rmtask]').forEach((b) => {
    b.onclick = async () => {
      const id = b.dataset.rmtask;
      if (!confirm(`Xoá hẳn task #${id} "${b.dataset.name}" trên CVAT?\n\n`
        + 'Mọi nhãn người đã sửa trong task này mất luôn và KHÔNG lấy lại được.')) return;
      if (!confirm(`Chắc chắn xoá task #${id}?`)) return;
      try {
        await api('/api/cvat/task/' + id, { method: 'DELETE' });
        toast(`Đã xoá task #${id}`, 'ok');
        refreshCvat();
      } catch (e) { toast(e.message, 'bad'); }
    };
  });

  el.querySelectorAll('[data-job]').forEach((b) => {
    b.onclick = async () => {
      try {
        await api(`/api/cvat/job/${b.dataset.job}/review`, {
          method: 'POST', body: JSON.stringify({ decision: b.dataset.dec }),
        });
        toast(`job ${b.dataset.job} → ${b.dataset.dec.toUpperCase()}`, 'ok');
        refreshCvat();
      } catch (e) { toast(e.message, 'bad'); }
    };
  });
}

// --- khởi động --------------------------------------------------------------

async function init() {
  buildNav();
  setupUpload();
  setupSourceForm();
  setupCanvas();
  setupLabels();
  setupParams();
  setupRun();
  try {
    await loadRegions();
    await loadLabels();
    await refreshState();
    await refreshInbox();
    refreshCvat();
  } catch (e) {
    toast('Không nạp được trạng thái: ' + e.message, 'bad');
  }
  setInterval(() => { if (!stream) refreshState().catch(() => {}); }, 10000);
}

init();
