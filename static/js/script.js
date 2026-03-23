// static/js/script.js
// BEV Occupancy Predictor — Frontend Logic

const API = '';
let currentBEV = { pred: null, gt: null, error: null };
let occChart   = null;

// ── Initialize ─────────────────────────────────────
window.onload = async () => {
  await loadHealth();
  await loadSamples();
  initTabs();
};

// ── Health check ───────────────────────────────────
async function loadHealth() {
  try {
    const res  = await fetch(`${API}/health`);
    const data = await res.json();
    document.getElementById('model-badge').textContent =
      `✓ Model ready | Epoch ${data.epoch} | `+
      `Best IoU: ${data.best_iou} | ${data.device}`;
  } catch {
    document.getElementById('model-badge').textContent =
      '⚠ Model loading...';
  }
}

// ── Load samples ───────────────────────────────────
async function loadSamples() {
  try {
    const res    = await fetch(`${API}/samples`);
    const data   = await res.json();
    const select = document.getElementById('sample-select');

    select.innerHTML = data.samples.map(s =>
      `<option value="${s.index}">
        #${s.index} — ${s.scene_name}
       </option>`
    ).join('');

    console.log(`Loaded ${data.total} samples`);

  } catch (e) {
    console.error('Failed to load samples:', e);
    document.getElementById('sample-select').innerHTML =
      '<option>Failed to load — check server</option>';
  }
}

// ── Predict sample ─────────────────────────────────
async function predictSample() {
  const btn = document.getElementById('btn-predict');
  const idx = document.getElementById('sample-select').value;

  // Loading state
  btn.disabled    = true;
  btn.innerHTML   = '<span class="spinner"></span> Predicting...';

  try {
    const fd = new FormData();
    fd.append('sample_idx', idx);

    const res  = await fetch(`${API}/predict/sample`, {
      method: 'POST',
      body  : fd
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();

    if (!data.success) {
      throw new Error('Prediction failed');
    }

    // Store BEV images
    currentBEV.pred  = data.bev_image;
    currentBEV.gt    = data.gt_image;
    currentBEV.error = data.error_image;

    // Show predicted BEV by default
    showBEVImage('pred');

    // Render cameras
    renderCameras(data.camera_grid);

    // Render metrics
    renderMetrics(data.metrics);

  } catch (e) {
    alert('Error: ' + e.message);
    console.error(e);
  } finally {
    btn.disabled  = false;
    btn.innerHTML = '🔍 Predict BEV';
  }
}

// ── Show BEV image ─────────────────────────────────
function showBEV(type, btnEl) {
  // Update tab active state
  document.querySelectorAll('.bev-tab').forEach(
    t => t.classList.remove('active')
  );
  if (btnEl) btnEl.classList.add('active');

  showBEVImage(type);
}

function showBEVImage(type) {
  const display = document.getElementById('bev-display');
  const img     = currentBEV[type];

  if (!img) {
    display.innerHTML =
      '<div class="bev-placeholder">Run prediction first</div>';
    return;
  }

  display.innerHTML = `
    <img
      src="data:image/png;base64,${img}"
      alt="BEV ${type}"
      class="fade-in"
    />`;
}

// ── Render camera grid ─────────────────────────────
function renderCameras(images) {
  const grid  = document.getElementById('cam-grid');
  const names = [
    'Front', 'Front Left', 'Front Right',
    'Back',  'Back Left',  'Back Right'
  ];

  grid.innerHTML = images.map((img, i) =>
    `<div class="cam-item fade-in">
       <img
         src="data:image/jpeg;base64,${img}"
         alt="${names[i]}"
       />
       <div class="cam-name">${names[i]}</div>
     </div>`
  ).join('');
}

// ── Render metrics ─────────────────────────────────
function renderMetrics(m) {
  // Update values
  document.getElementById('m-iou').textContent =
    m.occ_iou;
  document.getElementById('m-dwe').textContent =
    m.dwe;
  document.getElementById('m-occ').textContent =
    m.occupied_pct + '%';
  document.getElementById('m-time').textContent =
    m.inference_ms + 'ms';

  // Animate bars
  setTimeout(() => {
    document.getElementById('bar-iou').style.width =
      (m.occ_iou * 100) + '%';
    document.getElementById('bar-dwe').style.width =
      (m.dwe * 100) + '%';
  }, 100);

  // Render pie chart
  renderOccChart(m.occupied_pct, m.free_pct);
}

// ── Occupancy pie chart ────────────────────────────
function renderOccChart(occ, free) {
  const ctx = document.getElementById('occ-chart');
  if (!ctx) return;

  if (occChart) occChart.destroy();

  occChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels  : ['Occupied', 'Free'],
      datasets: [{
        data           : [occ, free],
        backgroundColor: ['#7c9aff', '#1b2432'],
        borderWidth    : 0,
        hoverOffset    : 4
      }]
    },
    options: {
      responsive: false,
      plugins   : {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx =>
              ` ${ctx.label}: ${ctx.parsed.toFixed(1)}%`
          }
        }
      },
      cutout: '72%',
      animation: { duration: 600 }
    }
  });
}

// ── Threshold slider ───────────────────────────────
function updateThreshold(val) {
  document.getElementById('thresh-val').textContent =
    parseFloat(val).toFixed(2);
}

// ── Video upload ───────────────────────────────────
async function handleVideoUpload(input) {
  const file = input.files[0];
  if (!file) return;

  const progress = document.getElementById('video-progress');
  const fill     = document.getElementById('progress-fill');
  const text     = document.getElementById('progress-text');
  const frames   = document.getElementById('video-frames');
  const gifOut   = document.getElementById('gif-output');

  // Reset
  frames.innerHTML       = '';
  gifOut.style.display   = 'none';
  progress.style.display = 'block';
  fill.style.width       = '10%';
  text.textContent       = `Uploading ${file.name}...`;

  try {
    const fd = new FormData();
    fd.append('video', file);

    fill.style.width = '40%';
    text.textContent = 'Processing frames on GPU...';

    const res  = await fetch(`${API}/predict/video`, {
      method: 'POST',
      body  : fd
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    fill.style.width = '90%';
    text.textContent = 'Rendering results...';

    const data = await res.json();

    fill.style.width = '100%';
    text.textContent =
      `✓ Done! ${data.total} frames processed`;

    // Show animated GIF
    if (data.gif_b64) {
      gifOut.style.display = 'block';
      document.getElementById('gif-img').src =
        `data:image/gif;base64,${data.gif_b64}`;
    }

    // Show side-by-side frames
    if (data.side_by_side && data.side_by_side.length) {
      frames.innerHTML = data.side_by_side.map((f, i) =>
        `<div class="frame-item fade-in">
           <img
             src="data:image/png;base64,${f}"
             alt="Frame ${i + 1}"
           />
           <div class="frame-label">Frame ${i + 1}</div>
         </div>`
      ).join('');
    }

  } catch (e) {
    fill.style.width = '100%';
    fill.style.background = 'var(--danger)';
    text.textContent = 'Error: ' + e.message;
    console.error(e);
  }
}

// ── Drag and drop ──────────────────────────────────
function handleDrop(e) {
  e.preventDefault();
  const file  = e.dataTransfer.files[0];
  if (!file || !file.type.startsWith('video/')) {
    alert('Please drop a video file');
    return;
  }
  const input   = document.getElementById('video-input');
  const dt      = new DataTransfer();
  dt.items.add(file);
  input.files   = dt.files;
  handleVideoUpload(input);
}

// ── Tab switching ──────────────────────────────────
function initTabs() {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(
        b => b.classList.remove('active')
      );
      document.querySelectorAll('.tab-panel').forEach(
        p => p.classList.remove('active')
      );
      btn.classList.add('active');
      const panel = document.getElementById(
        'tab-' + btn.dataset.tab
      );
      if (panel) panel.classList.add('active');
    });
  });
}