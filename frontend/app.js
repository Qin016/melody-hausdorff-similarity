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
const PLAYBACK_BPM = 140;
const BEAT_SECONDS = 60 / PLAYBACK_BPM;
const MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11];
const CHORD_PATTERN = [0, 5, 3, 4];
let generatedPointsUrl = null;
let audioContext = null;
let activeAudioNodes = [];
let playbackTimer = null;

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

function setPlaybackStatus(text) {
  $("playbackStatus").textContent = text;
}

function midiToFrequency(note) {
  return 440 * 2 ** ((note - 69) / 12);
}

async function ensureAudioContext() {
  const AudioEngine = window.AudioContext || window.webkitAudioContext;
  if (!AudioEngine) {
    throw new Error("Web Audio API is not supported");
  }
  audioContext ||= new AudioEngine();
  if (audioContext.state === "suspended") {
    await audioContext.resume();
  }
  return audioContext;
}

function parseGeneratedPoints(csvText) {
  const lines = csvText.trim().split(/\r?\n/);
  if (lines.length < 2) {
    return [];
  }

  const headers = lines[0].split(",");
  const pitchIndex = headers.indexOf("pitch_norm");
  const velocityIndex = headers.indexOf("velocity_norm");
  const midiPitchIndex = headers.indexOf("midi_pitch");
  const midiVelocityIndex = headers.indexOf("midi_velocity");
  const startBeatIndex = headers.indexOf("start_beat");
  const durationBeatIndex = headers.indexOf("duration_beat");
  const restIndex = headers.indexOf("rest");
  const phraseIndex = headers.indexOf("phrase_index");
  if (pitchIndex === -1 || velocityIndex === -1) {
    throw new Error("generated points CSV is missing pitch_norm or velocity_norm");
  }

  return lines
    .slice(1)
    .map((line) => line.split(","))
    .filter((cells) => cells.length > Math.max(pitchIndex, velocityIndex))
    .map((cells, index) => {
      const pitchNorm = Number(cells[pitchIndex]);
      const velocityNorm = Number(cells[velocityIndex]);
      const midi =
        midiPitchIndex === -1
          ? Math.min(84, Math.max(48, 55 + pitchNorm * 24))
          : Number(cells[midiPitchIndex]);
      const velocity =
        midiVelocityIndex === -1
          ? Math.min(120, Math.max(35, Math.round(45 + velocityNorm * 70)))
          : Number(cells[midiVelocityIndex]);
      const startBeat = startBeatIndex === -1 ? index * 0.5 : Number(cells[startBeatIndex]);
      const durationBeat = durationBeatIndex === -1 ? 0.5 : Number(cells[durationBeatIndex]);
      const rest = restIndex !== -1 && cells[restIndex].toLowerCase() === "true";
      const phrase = phraseIndex === -1 ? Math.floor(index / 16) + 1 : Number(cells[phraseIndex]);
      return { midi, velocity, startBeat, durationBeat, rest, phrase };
    });
}

function scaleMidi(root, degree, octaveOffset = 0) {
  return root + octaveOffset * 12 + MAJOR_SCALE[((degree % MAJOR_SCALE.length) + MAJOR_SCALE.length) % MAJOR_SCALE.length];
}

function createGuitarVoice(context, destination, midi, velocity, start, end, options = {}) {
  const detune = options.detune || 0;
  const level = options.level || 0.56;
  const filterFrequency = options.filterFrequency || 2600;
  const decay = options.decay || 0.28;
  const frequency = midiToFrequency(midi);
  const oscillator = context.createOscillator();
  const overtone = context.createOscillator();
  const gain = context.createGain();
  const overtoneGain = context.createGain();
  const filter = context.createBiquadFilter();

  oscillator.type = "triangle";
  overtone.type = "sawtooth";
  oscillator.frequency.setValueAtTime(frequency, start);
  overtone.frequency.setValueAtTime(frequency * 2.01, start);
  oscillator.detune.setValueAtTime(detune, start);
  overtone.detune.setValueAtTime(detune * 0.6, start);

  filter.type = "lowpass";
  filter.frequency.setValueAtTime(filterFrequency, start);
  filter.frequency.exponentialRampToValueAtTime(Math.max(700, filterFrequency * 0.38), end);
  filter.Q.setValueAtTime(1.1, start);

  const peak = (velocity / 127) * level;
  gain.gain.setValueAtTime(0.0001, start);
  gain.gain.exponentialRampToValueAtTime(peak, start + 0.012);
  gain.gain.exponentialRampToValueAtTime(Math.max(0.0001, peak * 0.18), start + decay);
  gain.gain.exponentialRampToValueAtTime(0.0001, end);
  overtoneGain.gain.setValueAtTime(peak * 0.16, start);
  overtoneGain.gain.exponentialRampToValueAtTime(0.0001, start + Math.min(0.18, decay));

  oscillator.connect(gain);
  overtone.connect(overtoneGain);
  overtoneGain.connect(gain);
  gain.connect(filter);
  filter.connect(destination);
  oscillator.start(start);
  overtone.start(start);
  oscillator.stop(end + 0.05);
  overtone.stop(end + 0.05);
  activeAudioNodes.push({ oscillator, gain, filter });
  activeAudioNodes.push({ oscillator: overtone, gain: overtoneGain, filter });
}

function stopGeneratedPlayback() {
  activeAudioNodes.forEach(({ oscillator }) => {
    try {
      oscillator.stop();
    } catch (error) {
      // Oscillators may already have ended naturally.
    }
  });
  activeAudioNodes = [];
  if (playbackTimer) {
    clearTimeout(playbackTimer);
    playbackTimer = null;
  }
  $("playGeneratedBtn").disabled = false;
  $("stopGeneratedBtn").disabled = true;
  setPlaybackStatus(generatedPointsUrl ? "可试听生成旋律" : "等待生成结果");
}

async function loadGeneratedNotes() {
  if (!generatedPointsUrl) {
    throw new Error("no generated melody is available");
  }
  const response = await fetch(generatedPointsUrl);
  if (!response.ok) {
    throw new Error(`generated points failed: ${response.status}`);
  }
  return parseGeneratedPoints(await response.text());
}

async function playGeneratedMelody() {
  stopGeneratedPlayback();
  $("playGeneratedBtn").disabled = true;
  $("stopGeneratedBtn").disabled = false;
  setPlaybackStatus("加载旋律中...");

  try {
    const context = await ensureAudioContext();
    const notes = await loadGeneratedNotes();
    if (!notes.length) {
      throw new Error("generated melody is empty");
    }

    const startAt = context.currentTime + 0.08;
    const totalBeats = Math.max(...notes.map((note) => note.startBeat + note.durationBeat));
    const endAt = startAt + totalBeats * BEAT_SECONDS;
    const master = context.createGain();
    master.gain.setValueAtTime(0.34, startAt);
    master.connect(context.destination);

    notes.forEach((note) => {
      if (note.rest) {
        return;
      }
      const noteStart = startAt + note.startBeat * BEAT_SECONDS;
      const noteEnd = noteStart + Math.max(0.18, note.durationBeat * BEAT_SECONDS * 1.04);
      createGuitarVoice(context, master, note.midi, note.velocity, noteStart, noteEnd, {
        detune: 0,
        level: 0.64,
        filterFrequency: 3000,
        decay: Math.min(0.42, Math.max(0.2, note.durationBeat * BEAT_SECONDS * 0.58)),
      });
    });

    const phraseBeats = 4;
    const chordCount = Math.ceil(totalBeats / phraseBeats);
    for (let chordIndex = 0; chordIndex < chordCount; chordIndex += 1) {
      const degree = CHORD_PATTERN[chordIndex % CHORD_PATTERN.length];
      const chordStart = startAt + chordIndex * phraseBeats * BEAT_SECONDS;
      const chordEnd = chordStart + phraseBeats * BEAT_SECONDS * 0.94;
      [degree, degree + 2, degree + 4].forEach((chordDegree) => {
        createGuitarVoice(context, master, scaleMidi(48, chordDegree), 46, chordStart, chordEnd, {
          detune: -4 + chordDegree * 1.5,
          level: 0.2,
          filterFrequency: 1900,
          decay: 0.7,
        });
      });
    }

    const duration = totalBeats * BEAT_SECONDS;
    setPlaybackStatus(`播放中 · ${notes.length} 个音符`);
    playbackTimer = setTimeout(stopGeneratedPlayback, (duration + 0.4) * 1000);
  } catch (error) {
    console.error(error);
    stopGeneratedPlayback();
    setPlaybackStatus("播放失败，请先生成旋律");
  }
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
  generatedPointsUrl = withApi("/artifacts/data/processed/interpolated_melody_points.csv");
  $("pointsLink").href = generatedPointsUrl;
  setPlaybackStatus("可试听生成旋律");
}

function renderSongOption(song) {
  return `<option value="${song.song_id}">${song.song_id} · ${song.genre} · ${song.file_name}</option>`;
}

function setSelectValueIfExists(select, value) {
  if ([...select.options].some((option) => option.value === value)) {
    select.value = value;
  }
}

async function loadSongs() {
  const songs = await api("/api/songs");
  const options = songs.map(renderSongOption).join("");
  $("songSelect").innerHTML = options;
  $("songSelect").value = "11";

  const autoOption = '<option value="">自动选择相似曲对</option>';
  $("generateSongA").innerHTML = autoOption + options;
  $("generateSongB").innerHTML = autoOption + options;
  setSelectValueIfExists($("generateSongA"), "30");
  setSelectValueIfExists($("generateSongB"), "69");
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
    const smoothWindow = Number($("smoothInput").value);
    const minDistance = Number($("distanceInput").value);
    const songA = $("generateSongA").value;
    const songB = $("generateSongB").value;
    if ((songA && !songB) || (!songA && songB)) {
      setPlaybackStatus("请选择两首歌，或都设为自动选择");
      return;
    }
    if (songA && songA === songB) {
      setPlaybackStatus("源曲 A 和源曲 B 不能相同");
      return;
    }

    const payload = {
      alpha,
      note_count: noteCount,
      smooth_window: smoothWindow,
      min_distance: minDistance,
    };
    if (songA && songB) {
      payload.song_a = Number(songA);
      payload.song_b = Number(songB);
    }

    const data = await api("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    $("generateImg").src = withApi(`${data.figure}?t=${Date.now()}`);
    $("midiLink").href = withApi(`${data.midi}?t=${Date.now()}`);
    generatedPointsUrl = withApi(`${data.points_csv}?t=${Date.now()}`);
    $("pointsLink").href = generatedPointsUrl;
    const hasSelectedPair = data.song_a !== null && data.song_b !== null;
    const pairText = hasSelectedPair ? `源曲 ${data.song_a} → ${data.song_b}` : "自动曲对";
    setPlaybackStatus(`${pairText} 已生成，可试听`);
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
$("playGeneratedBtn").addEventListener("click", playGeneratedMelody);
$("stopGeneratedBtn").addEventListener("click", stopGeneratedPlayback);
$("alphaInput").addEventListener("input", () => {
  $("alphaValue").textContent = Number($("alphaInput").value).toFixed(2);
});
$("smoothInput").addEventListener("input", () => {
  $("smoothValue").textContent = $("smoothInput").value;
});
$("distanceInput").addEventListener("input", () => {
  $("distanceValue").textContent = Number($("distanceInput").value).toFixed(2);
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
