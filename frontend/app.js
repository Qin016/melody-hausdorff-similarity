const API = "http://127.0.0.1:5000";
const genreColors = new Map([
  ["Country", "#1f77b4"],
  ["Jazz", "#ff9f1c"],
  ["Modern Pop", "#2ec4b6"],
  ["Rap", "#e71d36"],
  ["Rhythm and Blues", "#8e5cf4"],
  ["Rock", "#9c6644"],
  ["Western Classical", "#ff4fb8"],
  ["Western Folk", "#8d99ae"],
  ["Worldbeat", "#d4d700"],
]);

const $ = (id) => document.getElementById(id);

function withApi(path) {
  return `${API}${path}`;
}

async function api(path, options = {}) {
  const response = await fetch(withApi(path), options);
  if (!response.ok) {
    throw new Error(`API ${path} failed: ${response.status}`);
  }
  return response.json();
}

function setStatus(text) {
  $("apiStatus").textContent = text;
}

function formatNumber(value, digits = 0) {
  return Number(value).toLocaleString("zh-CN", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
}

function metric(label, value) {
  return `<div class="metric"><strong>${value}</strong><span>${label}</span></div>`;
}

function renderSummary(data) {
  const dataset = data.dataset;
  const hausdorff = data.hausdorff;
  $("metrics").innerHTML = [
    metric("MIDI 文件", formatNumber(dataset.midi_files)),
    metric("曲风类别", formatNumber(dataset.genres)),
    metric("平衡曲目", formatNumber(dataset.balanced_songs)),
    metric("采样旋律点", formatNumber(dataset.sampled_points)),
    metric("1-NN 准确率", `${formatNumber(hausdorff.one_nn_accuracy * 100, 2)}%`),
  ].join("");

  const maxCount = Math.max(...dataset.label_summary.map((row) => row.file_count));
  $("genreBars").innerHTML = dataset.label_summary
    .map((row) => {
      const width = (row.file_count / maxCount) * 100;
      return `
        <div class="bar">
          <span>${row.genre}</span>
          <div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div>
          <strong>${row.file_count}</strong>
        </div>
      `;
    })
    .join("");

  const artifacts = data.artifacts;
  $("heatmapImg").src = withApi(artifacts.distance_heatmap);
  $("boxplotImg").src = withApi(artifacts.same_diff_boxplot);
  $("singleCurveImg").src = withApi(artifacts.single_curve);
  $("genreCurveImg").src = withApi(artifacts.genre_curves);
  $("curveFrame").src = withApi(artifacts.interactive_curves);
  $("generateImg").src = withApi(artifacts.interpolation);
  $("midiLink").href = withApi(artifacts.generated_midi);
}

async function loadSongs() {
  const songs = await api("/api/songs");
  $("songSelect").innerHTML = songs
    .map((song) => `<option value="${song.song_id}">${song.song_id} · ${song.genre} · ${song.file_name}</option>`)
    .join("");
  $("songSelect").value = "11";
}

async function runSearch() {
  const button = $("searchBtn");
  button.disabled = true;
  button.textContent = "检索中...";
  try {
    const songId = $("songSelect").value;
    const topK = $("topKInput").value;
    const data = await api(`/api/search?song_id=${songId}&top_k=${topK}`);
    $("searchRows").innerHTML = data.results
      .map(
        (row) => `
          <tr>
            <td>${row.rank}</td>
            <td>${row.genre}</td>
            <td>${row.file_name}</td>
            <td>${Number(row.hausdorff_distance).toFixed(4)}</td>
          </tr>
        `,
      )
      .join("");
    $("searchImg").src = withApi(`${data.figure}?t=${Date.now()}`);
  } finally {
    button.disabled = false;
    button.textContent = "开始检索";
  }
}

function drawMusicMap(points) {
  const canvas = $("mapCanvas");
  const ctx = canvas.getContext("2d");
  const tooltip = $("mapTooltip");
  const w = canvas.width;
  const h = canvas.height;
  const pad = 46;
  const xs = points.map((p) => Number(p.x));
  const ys = points.map((p) => Number(p.y));
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const plotted = points.map((p) => {
    const x = pad + ((Number(p.x) - minX) / (maxX - minX || 1)) * (w - pad * 2);
    const y = h - pad - ((Number(p.y) - minY) / (maxY - minY || 1)) * (h - pad * 2);
    return { ...p, px: x, py: y };
  });

  function paint(highlight = null) {
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = "#e5e7eb";
    ctx.lineWidth = 1;
    for (let i = 0; i < 8; i += 1) {
      const x = pad + (i / 7) * (w - pad * 2);
      const y = pad + (i / 7) * (h - pad * 2);
      ctx.beginPath();
      ctx.moveTo(x, pad);
      ctx.lineTo(x, h - pad);
      ctx.moveTo(pad, y);
      ctx.lineTo(w - pad, y);
      ctx.stroke();
    }
    plotted.forEach((p) => {
      ctx.beginPath();
      ctx.fillStyle = genreColors.get(p.genre) || "#64748b";
      ctx.globalAlpha = highlight && highlight !== p ? 0.25 : 0.82;
      ctx.arc(p.px, p.py, highlight === p ? 7 : 5, 0, Math.PI * 2);
      ctx.fill();
    });
    ctx.globalAlpha = 1;
  }

  canvas.onmousemove = (event) => {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const x = (event.clientX - rect.left) * scaleX;
    const y = (event.clientY - rect.top) * scaleY;
    const hit = plotted.find((p) => Math.hypot(p.px - x, p.py - y) < 8);
    paint(hit);
    if (hit) {
      tooltip.style.display = "block";
      tooltip.style.left = `${event.clientX - rect.left + 14}px`;
      tooltip.style.top = `${event.clientY - rect.top + 14}px`;
      tooltip.innerHTML = `<strong>${hit.genre}</strong><br>${hit.file_name}<br>song_id=${hit.song_id}`;
    } else {
      tooltip.style.display = "none";
    }
  };
  canvas.onmouseleave = () => {
    tooltip.style.display = "none";
    paint();
  };
  paint();
}

async function loadMusicMap() {
  const data = await api("/api/music-map");
  drawMusicMap(data.points);
  $("mapHtmlLink").href = withApi(data.interactive);
}

async function generateMelody() {
  const button = $("generateBtn");
  button.disabled = true;
  button.textContent = "生成中...";
  try {
    const alpha = Number($("alphaInput").value);
    const noteCount = Number($("noteCountInput").value);
    const data = await api("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ alpha, note_count: noteCount }),
    });
    $("generateImg").src = withApi(`${data.figure}?t=${Date.now()}`);
    $("midiLink").href = withApi(`${data.midi}?t=${Date.now()}`);
    $("pointsLink").href = withApi(`${data.points_csv}?t=${Date.now()}`);
  } finally {
    button.disabled = false;
    button.textContent = "生成旋律";
  }
}

async function bootstrap() {
  try {
    setStatus("API 连接中...");
    const summary = await api("/api/summary");
    renderSummary(summary);
    await loadSongs();
    await loadMusicMap();
    await loadExistingSearch();
    setStatus("API 已连接 · 数据加载完成");
  } catch (error) {
    console.error(error);
    setStatus("API 连接失败，请先启动 Flask 后端");
  }
}

$("refreshBtn").addEventListener("click", bootstrap);
$("searchBtn").addEventListener("click", runSearch);
$("generateBtn").addEventListener("click", generateMelody);
$("alphaInput").addEventListener("input", () => {
  $("alphaValue").textContent = Number($("alphaInput").value).toFixed(2);
});

bootstrap();

async function loadExistingSearch() {
  try {
    const data = await api("/api/search?song_id=11&top_k=8");
    $("searchRows").innerHTML = data.results
      .map(
        (row) => `
          <tr>
            <td>${row.rank}</td>
            <td>${row.genre}</td>
            <td>${row.file_name}</td>
            <td>${Number(row.hausdorff_distance).toFixed(4)}</td>
          </tr>
        `,
      )
      .join("");
    $("searchImg").src = withApi(data.figure);
  } catch (error) {
    console.warn("No existing search result", error);
  }
}
