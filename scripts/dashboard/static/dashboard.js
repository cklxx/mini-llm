async function fetchJSON(url, options = {}) {
  const res = await fetch(url, options);
  if (!res.ok) {
    throw new Error(`请求失败: ${res.status}`);
  }
  return res.json();
}

function fmtBytes(bytes) {
  if (!bytes && bytes !== 0) return '未知';
  const units = ['B', 'KB', 'MB', 'GB'];
  let size = bytes;
  let idx = 0;
  while (size >= 1024 && idx < units.length - 1) {
    size /= 1024;
    idx += 1;
  }
  return `${size.toFixed(size < 10 ? 1 : 0)} ${units[idx]}`;
}

function fmtLines(n, capped) {
  if (!n) return '未知';
  return capped ? `${n}+` : `${n}`;
}

function renderOverview(data) {
  const container = document.getElementById('overview');
  container.innerHTML = '';
  const cards = [
    { label: '数据集', value: data.datasets, accent: '#d1d5db' },
    { label: '运行记录', value: data.runs, accent: '#cbd5e1' },
    { label: '配置模板', value: data.configs, accent: '#e5e7eb' },
  ];
  cards.forEach((item) => {
    const div = document.createElement('div');
    div.className = 'card';
    div.innerHTML = `
      <h3>${item.label}</h3>
      <div class="stat" style="color:${item.accent}">${item.value ?? 0}</div>
      <div class="muted">实时统计</div>
    `;
    container.appendChild(div);
  });

  const tag = document.getElementById('latest-run-tag');
  if (data.latest_run) {
    const run = data.latest_run;
    tag.textContent = `最近运行 · ${run.name} (${run.stage})`;
  } else {
    tag.textContent = '最近运行：暂无记录';
  }
}

function renderDatasets(rows) {
  const tbody = document.querySelector('#dataset-table tbody');
  tbody.innerHTML = '';
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="5" class="muted">未找到数据文件</td></tr>';
    return;
  }
  rows.forEach((row) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><input type="checkbox" data-path="${row.path}"></td>
      <td><div style="font-family:monospace; font-size:13px;">${row.path}</div></td>
      <td>${fmtLines(row.line_count, row.line_count_capped)}</td>
      <td>${fmtBytes(row.size_bytes)}</td>
      <td class="muted">${row.preview.slice(0, 2).map((p) => p.replace(/</g, '&lt;')).join('<br>')}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderConfigs(configs) {
  const grid = document.getElementById('config-grid');
  grid.innerHTML = '';
  if (!configs.length) {
    grid.innerHTML = '<div class="card muted">暂无配置模板</div>';
    return;
  }
  configs.forEach((cfg) => {
    const div = document.createElement('div');
    div.className = 'card';
    const meta = cfg.content.meta || {};
    const stage = meta.stage || 'pipeline';
    div.innerHTML = `
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <h3>${cfg.name}</h3>
        <span class="tag">${stage}</span>
      </div>
      <div class="muted" style="margin-bottom:8px;">版本：${cfg.version || '未标注'}</div>
      <div class="muted" style="font-size:13px; line-height:1.6;">${cfg.description || '配置文件已准备好，可用于 run.sh 或 mlx 训练脚本。'}</div>
      <div class="muted" style="margin-top:8px; font-family:monospace;">${cfg.path}</div>
    `;
    grid.appendChild(div);
  });
}

function grayscalePalette(index) {
  const shades = ['#e5e7eb', '#cbd5e1', '#94a3b8', '#6b7280'];
  return shades[index % shades.length];
}

async function drawChart(canvasId, scalars) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;
  const datasets = Object.entries(scalars)
    .filter(([, values]) => values && values.length)
    .map(([tag, values]) => ({
      label: tag,
      data: values.map((v) => ({ x: v.step, y: v.value })),
      fill: false,
      borderWidth: 2,
      borderColor: grayscalePalette(Object.keys(scalars).indexOf(tag)),
      backgroundColor: 'rgba(229, 231, 235, 0.08)',
    }));

  if (!datasets.length) {
    ctx.replaceWith(Object.assign(document.createElement('div'), { className: 'muted', textContent: '暂无 TensorBoard 标量' }));
    return;
  }

  new Chart(ctx, {
    type: 'line',
    data: { datasets },
    options: {
      responsive: true,
      plugins: { legend: { display: true, labels: { color: '#cbd5e1' } } },
      scales: {
        x: { type: 'linear', ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.08)' } },
        y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.08)' } },
      },
    },
  });
}

async function renderRuns(runs) {
  const container = document.getElementById('runs');
  container.innerHTML = '';
  if (!runs.length) {
    container.innerHTML = '<div class="card muted">暂无运行记录</div>';
    return;
  }

  for (let i = 0; i < runs.length; i += 1) {
    const run = runs[i];
    const card = document.createElement('div');
    card.className = 'card run-card';
    const metricLines = Object.entries(run.metrics || {})
      .map(([k, v]) => `<span class="tag">${k}: ${v.toFixed(4)}</span>`)
      .join(' ');
    card.innerHTML = `
      <div>
        <div style="display:flex; align-items:center; gap:8px;">
          <h3 style="margin:0;">${run.name}</h3>
          <span class="tag">${run.stage}</span>
        </div>
        <div class="muted" style="margin:4px 0;">最近修改：${new Date(run.modified_at).toLocaleString()}</div>
        <div class="muted" style="margin:4px 0;">${run.latest_checkpoint ? `Checkpoint: ${run.latest_checkpoint}` : '暂无权重文件'}</div>
        <div style="display:flex; gap:6px; flex-wrap:wrap;">${metricLines || '<span class="muted">暂无标量</span>'}</div>
      </div>
      <div><canvas id="chart-${i}" height="120"></canvas></div>
    `;
    container.appendChild(card);

    try {
      const scalars = await fetchJSON(`/api/runs/${encodeURIComponent(run.id)}/scalars`);
      await drawChart(`chart-${i}`, scalars.scalars || {});
    } catch (err) {
      console.error('加载标量失败', err);
    }
  }
}

async function refresh() {
  const [overview, datasets, configs, runs] = await Promise.all([
    fetchJSON('/api/overview'),
    fetchJSON('/api/datasets'),
    fetchJSON('/api/configs'),
    fetchJSON('/api/runs'),
  ]);
  renderOverview(overview);
  renderDatasets(datasets);
  renderConfigs(configs);
  renderRuns(runs);
}

async function setupSnapshotBuilder() {
  const btn = document.getElementById('build-snapshot');
  btn.addEventListener('click', async () => {
    const name = document.getElementById('snapshot-name').value.trim();
    const selected = Array.from(document.querySelectorAll('#dataset-table input[type="checkbox"]:checked'))
      .map((el) => el.dataset.path);
    if (!selected.length) {
      alert('请至少选择一个数据文件');
      return;
    }
    btn.disabled = true;
    btn.textContent = '生成中...';
    try {
      const res = await fetchJSON('/api/datasets/materialize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, files: selected }),
      });
      alert(`快照已生成：${res.combined_path}\n行数：${res.line_count}`);
      await refresh();
    } catch (err) {
      alert(`生成失败：${err.message}`);
    } finally {
      btn.disabled = false;
      btn.textContent = '生成快照';
    }
  });
}

document.addEventListener('DOMContentLoaded', async () => {
  await refresh();
  setupSnapshotBuilder();
});
