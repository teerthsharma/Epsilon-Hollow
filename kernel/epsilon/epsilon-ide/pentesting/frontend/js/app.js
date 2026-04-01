/**
 * sealMega IDE — Main Application
 * Standalone web-based IDE with Monaco Editor, file explorer, terminal, and AI engine panel.
 */

const API_BASE = window.location.origin;
let monacoEditor = null;
let openFiles = new Map(); // filepath -> { model, content, modified }
let activeFile = null;
let terminal = null;
let terminalWs = null;

// ─── Boot ───
document.addEventListener('DOMContentLoaded', () => {
    initActivityBar();
    initSidebarResize();
    initPanelResize();
    initPanelTabs();
    initTerminal();
    initMonaco();
    initFileExplorer();
    initAIPanel();
    initKeyBindings();
    pollStatus();
});

// ═══════════════════════════════════════
// Activity Bar
// ═══════════════════════════════════════
function initActivityBar() {
    document.querySelectorAll('.activity-btn[data-panel]').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.activity-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const panelId = 'panel-' + btn.dataset.panel;
            document.querySelectorAll('.sidebar-panel').forEach(p => p.classList.remove('active'));
            const panel = document.getElementById(panelId);
            if (panel) panel.classList.add('active');
        });
    });
}

// ═══════════════════════════════════════
// Sidebar Resize
// ═══════════════════════════════════════
function initSidebarResize() {
    const handle = document.getElementById('sidebar-resize');
    const sidebar = document.getElementById('sidebar');
    let startX, startW;
    handle.addEventListener('mousedown', e => {
        startX = e.clientX;
        startW = sidebar.offsetWidth;
        handle.classList.add('dragging');
        const onMove = e2 => {
            const w = Math.max(160, Math.min(500, startW + e2.clientX - startX));
            sidebar.style.width = w + 'px';
        };
        const onUp = () => {
            handle.classList.remove('dragging');
            document.removeEventListener('mousemove', onMove);
            document.removeEventListener('mouseup', onUp);
        };
        document.addEventListener('mousemove', onMove);
        document.addEventListener('mouseup', onUp);
    });
}

// ═══════════════════════════════════════
// Panel Resize (Terminal Height)
// ═══════════════════════════════════════
function initPanelResize() {
    const handle = document.getElementById('panel-resize');
    const panel = document.getElementById('bottom-panel');
    let startY, startH;
    handle.addEventListener('mousedown', e => {
        startY = e.clientY;
        startH = panel.offsetHeight;
        handle.classList.add('dragging');
        const onMove = e2 => {
            const h = Math.max(80, Math.min(500, startH - (e2.clientY - startY)));
            panel.style.height = h + 'px';
        };
        const onUp = () => {
            handle.classList.remove('dragging');
            document.removeEventListener('mousemove', onMove);
            document.removeEventListener('mouseup', onUp);
            if (terminal) terminal.fit && terminal.fit();
        };
        document.addEventListener('mousemove', onMove);
        document.addEventListener('mouseup', onUp);
    });

    document.getElementById('btn-toggle-panel').addEventListener('click', () => {
        panel.style.display = panel.style.display === 'none' ? 'flex' : 'none';
    });
}

// ═══════════════════════════════════════
// Panel Tabs
// ═══════════════════════════════════════
function initPanelTabs() {
    document.querySelectorAll('.panel-tab[data-target]').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.panel-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            document.querySelectorAll('.panel-content').forEach(p => p.classList.remove('active'));
            const target = document.getElementById(tab.dataset.target);
            if (target) target.classList.add('active');
        });
    });
}

// ═══════════════════════════════════════
// Terminal (xterm.js)
// ═══════════════════════════════════════
function initTerminal() {
    if (typeof Terminal === 'undefined') return;
    terminal = new Terminal({
        fontSize: 13,
        fontFamily: "'Cascadia Code', 'Fira Code', Consolas, monospace",
        theme: {
            background: '#1e1e1e',
            foreground: '#cccccc',
            cursor: '#aeafad',
            selectionBackground: '#264f78',
            black: '#1e1e1e',
            red: '#f44747',
            green: '#6a9955',
            yellow: '#dcdcaa',
            blue: '#569cd6',
            magenta: '#c586c0',
            cyan: '#4ec9b0',
            white: '#d4d4d4',
        },
        cursorBlink: true,
        allowProposedApi: true,
    });

    const fitAddon = new FitAddon.FitAddon();
    terminal.loadAddon(fitAddon);
    terminal.open(document.getElementById('terminal'));
    fitAddon.fit();
    terminal.fit = () => fitAddon.fit();

    terminal.writeln('\x1b[1;36m  sealMega IDE Terminal\x1b[0m');
    terminal.writeln('\x1b[90m  Type commands below. Connected to backend shell.\x1b[0m');
    terminal.writeln('');

    // Connect to WebSocket terminal
    connectTerminalWs();

    window.addEventListener('resize', () => fitAddon.fit());
}

function connectTerminalWs() {
    const wsUrl = API_BASE.replace('http', 'ws') + '/ws/terminal';
    try {
        terminalWs = new WebSocket(wsUrl);
        terminalWs.onopen = () => {
            terminal.writeln('\x1b[32m  Connected to shell.\x1b[0m\r\n');
        };
        terminalWs.onmessage = (event) => {
            terminal.write(event.data);
        };
        terminalWs.onclose = () => {
            terminal.writeln('\r\n\x1b[31m  Shell disconnected.\x1b[0m');
        };
        terminalWs.onerror = () => {
            terminal.writeln('\x1b[33m  Shell not available. Using local echo.\x1b[0m\r\n');
            setupLocalEcho();
        };
        terminal.onData(data => {
            if (terminalWs && terminalWs.readyState === WebSocket.OPEN) {
                terminalWs.send(data);
            }
        });
    } catch {
        setupLocalEcho();
    }
}

function setupLocalEcho() {
    let currentLine = '';
    const prompt = '\x1b[36msealMega\x1b[0m \x1b[33m>\x1b[0m ';
    terminal.write(prompt);
    terminal.onData(data => {
        if (data === '\r') {
            terminal.writeln('');
            if (currentLine.trim()) {
                executeLocalCommand(currentLine.trim());
            }
            currentLine = '';
            terminal.write(prompt);
        } else if (data === '\x7f') {
            if (currentLine.length > 0) {
                currentLine = currentLine.slice(0, -1);
                terminal.write('\b \b');
            }
        } else if (data >= ' ') {
            currentLine += data;
            terminal.write(data);
        }
    });
}

async function executeLocalCommand(cmd) {
    try {
        const res = await fetch(`${API_BASE}/api/v1/claw/execute`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ command: cmd, cwd: '.' }),
        });
        const data = await res.json();
        if (!data.approved) {
            terminal.writeln(`\x1b[31mBlocked: ${data.safetyReason}\x1b[0m`);
        } else {
            if (data.output) terminal.write(data.output.replace(/\n/g, '\r\n'));
            if (data.exitCode !== 0) {
                terminal.writeln(`\x1b[31m[exit ${data.exitCode}]\x1b[0m`);
            }
        }
    } catch {
        terminal.writeln(`\x1b[31mBackend not reachable.\x1b[0m`);
    }
}

// ═══════════════════════════════════════
// Monaco Editor
// ═══════════════════════════════════════
function initMonaco() {
    require(['vs/editor/editor.main'], function () {
        monaco.editor.defineTheme('sealMegaDark', {
            base: 'vs-dark',
            inherit: true,
            rules: [],
            colors: {
                'editor.background': '#1e1e1e',
                'editorGutter.background': '#1e1e1e',
                'editor.lineHighlightBackground': '#2a2d2e',
            }
        });
        monaco.editor.setTheme('sealMegaDark');
        console.log('[sealMega] Monaco Editor initialized.');
    });
}

function openFileInEditor(filePath, content) {
    // Hide welcome screen
    document.getElementById('welcome-screen').classList.remove('active');
    const monacoContainer = document.getElementById('monaco-container');
    monacoContainer.classList.add('active');

    // Get or create Monaco editor
    if (!monacoEditor) {
        monacoEditor = monaco.editor.create(monacoContainer, {
            value: content,
            language: getLanguageFromPath(filePath),
            theme: 'sealMegaDark',
            automaticLayout: true,
            minimap: { enabled: true },
            fontSize: 14,
            fontFamily: "'Cascadia Code', 'Fira Code', Consolas, monospace",
            fontLigatures: true,
            scrollBeyondLastLine: false,
            renderWhitespace: 'selection',
            cursorBlinking: 'smooth',
            smoothScrolling: true,
            padding: { top: 8 },
        });

        // Update status bar on cursor move
        monacoEditor.onDidChangeCursorPosition(e => {
            document.getElementById('status-line').textContent =
                `Ln ${e.position.lineNumber}, Col ${e.position.column}`;
        });

        // Track modifications
        monacoEditor.onDidChangeModelContent(() => {
            if (activeFile) {
                const tab = document.querySelector(`.tab[data-file="${CSS.escape(activeFile)}"]`);
                if (tab) {
                    const nameEl = tab.querySelector('.tab-name');
                    if (!nameEl.textContent.startsWith('● ')) {
                        nameEl.textContent = '● ' + nameEl.textContent;
                    }
                }
            }
        });
    }

    // Create or switch model
    if (!openFiles.has(filePath)) {
        const ext = filePath.split('.').pop();
        const lang = getLanguageFromPath(filePath);
        const model = monaco.editor.createModel(content, lang, monaco.Uri.file(filePath));
        openFiles.set(filePath, { model, content });
        addEditorTab(filePath);
    }

    monacoEditor.setModel(openFiles.get(filePath).model);
    activeFile = filePath;
    activateTab(filePath);

    // Update title bar and status
    document.getElementById('titlebar-path').textContent = filePath;
    document.getElementById('status-language').textContent = getLanguageFromPath(filePath);
}

function getLanguageFromPath(path) {
    const ext = path.split('.').pop().toLowerCase();
    const map = {
        'js': 'javascript', 'jsx': 'javascript', 'ts': 'typescript', 'tsx': 'typescript',
        'py': 'python', 'rs': 'rust', 'go': 'go', 'java': 'java', 'c': 'c', 'cpp': 'cpp',
        'h': 'c', 'cs': 'csharp', 'rb': 'ruby', 'php': 'php', 'swift': 'swift',
        'kt': 'kotlin', 'html': 'html', 'css': 'css', 'scss': 'scss', 'less': 'less',
        'json': 'json', 'xml': 'xml', 'yaml': 'yaml', 'yml': 'yaml', 'md': 'markdown',
        'sql': 'sql', 'sh': 'shell', 'bash': 'shell', 'ps1': 'powershell',
        'dockerfile': 'dockerfile', 'toml': 'ini', 'ini': 'ini', 'txt': 'plaintext',
    };
    return map[ext] || 'plaintext';
}

// ═══════════════════════════════════════
// Editor Tabs
// ═══════════════════════════════════════
function addEditorTab(filePath) {
    const tabs = document.getElementById('editor-tabs');
    const fileName = filePath.split(/[/\\]/).pop();

    const tab = document.createElement('div');
    tab.className = 'tab';
    tab.dataset.file = filePath;
    tab.innerHTML = `
        <span class="tab-icon">${getFileIcon(fileName)}</span>
        <span class="tab-name">${fileName}</span>
        <button class="tab-close">✕</button>
    `;

    tab.addEventListener('click', (e) => {
        if (e.target.classList.contains('tab-close')) {
            closeTab(filePath);
            return;
        }
        switchToFile(filePath);
    });

    tab.querySelector('.tab-close').addEventListener('click', (e) => {
        e.stopPropagation();
        closeTab(filePath);
    });

    tabs.appendChild(tab);
}

function activateTab(filePath) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    const tab = document.querySelector(`.tab[data-file="${CSS.escape(filePath)}"]`);
    if (tab) tab.classList.add('active');
}

function switchToFile(filePath) {
    if (!openFiles.has(filePath)) return;
    monacoEditor.setModel(openFiles.get(filePath).model);
    activeFile = filePath;
    activateTab(filePath);
    document.getElementById('titlebar-path').textContent = filePath;
    document.getElementById('status-language').textContent = getLanguageFromPath(filePath);
}

function closeTab(filePath) {
    const tab = document.querySelector(`.tab[data-file="${CSS.escape(filePath)}"]`);
    if (tab) tab.remove();
    const fileData = openFiles.get(filePath);
    if (fileData) {
        fileData.model.dispose();
        openFiles.delete(filePath);
    }
    if (activeFile === filePath) {
        const remaining = Array.from(openFiles.keys());
        if (remaining.length > 0) {
            switchToFile(remaining[remaining.length - 1]);
        } else {
            activeFile = null;
            document.getElementById('welcome-screen').classList.add('active');
            document.getElementById('monaco-container').classList.remove('active');
            document.getElementById('titlebar-path').textContent = 'Welcome';
        }
    }
}

function getFileIcon(name) {
    const ext = name.split('.').pop().toLowerCase();
    const icons = {
        'py': '🐍', 'js': '📜', 'ts': '💠', 'jsx': '⚛️', 'tsx': '⚛️',
        'html': '🌐', 'css': '🎨', 'json': '📋', 'md': '📝', 'rs': '🦀',
        'go': '🐹', 'java': '☕', 'rb': '💎', 'php': '🐘', 'sql': '🗃️',
        'sh': '🐚', 'yml': '⚙️', 'yaml': '⚙️', 'toml': '⚙️', 'txt': '📄',
        'svg': '🖼️', 'png': '🖼️', 'jpg': '🖼️',
    };
    return icons[ext] || '📄';
}

// ═══════════════════════════════════════
// File Explorer
// ═══════════════════════════════════════
async function initFileExplorer() {
    await loadFileTree('/');
    document.getElementById('btn-refresh').addEventListener('click', () => loadFileTree('/'));
}

async function loadFileTree(path) {
    const tree = document.getElementById('file-tree');
    tree.innerHTML = '<div class="panel-placeholder">Loading...</div>';

    try {
        const res = await fetch(`${API_BASE}/api/v1/files/list?path=${encodeURIComponent(path)}`);
        if (!res.ok) throw new Error('Failed to list files');
        const data = await res.json();
        tree.innerHTML = '';
        renderFileTree(tree, data.entries, 0);
    } catch {
        tree.innerHTML = '<div class="panel-placeholder">Cannot load files.<br>Is the backend running?</div>';
    }
}

function renderFileTree(container, entries, depth) {
    // Sort: directories first, then files
    entries.sort((a, b) => {
        if (a.isDir && !b.isDir) return -1;
        if (!a.isDir && b.isDir) return 1;
        return a.name.localeCompare(b.name);
    });

    entries.forEach(entry => {
        const item = document.createElement('div');
        item.className = 'file-item' + (entry.isDir ? ' dir' : '');
        item.innerHTML = `
            <span class="indent" style="width:${depth * 16}px"></span>
            ${entry.isDir ? '<span class="arrow">▶</span>' : '<span style="width:16px"></span>'}
            <span class="icon">${entry.isDir ? '📁' : getFileIcon(entry.name)}</span>
            <span class="name">${entry.name}</span>
        `;

        if (entry.isDir) {
            let loaded = false;
            let expanded = false;
            const childContainer = document.createElement('div');
            childContainer.style.display = 'none';

            item.addEventListener('click', async () => {
                expanded = !expanded;
                const arrow = item.querySelector('.arrow');
                arrow.classList.toggle('open', expanded);
                item.querySelector('.icon').textContent = expanded ? '📂' : '📁';

                if (!loaded) {
                    try {
                        const res = await fetch(`${API_BASE}/api/v1/files/list?path=${encodeURIComponent(entry.path)}`);
                        const data = await res.json();
                        renderFileTree(childContainer, data.entries, depth + 1);
                        loaded = true;
                    } catch {
                        childContainer.innerHTML = '<div class="panel-placeholder" style="padding:4px 0 4px 32px;font-size:11px;">Error loading</div>';
                    }
                }
                childContainer.style.display = expanded ? 'block' : 'none';
            });

            container.appendChild(item);
            container.appendChild(childContainer);
        } else {
            item.addEventListener('click', async () => {
                // Select visual
                document.querySelectorAll('.file-item').forEach(f => f.classList.remove('selected'));
                item.classList.add('selected');
                // Open file
                try {
                    const res = await fetch(`${API_BASE}/api/v1/files/read?path=${encodeURIComponent(entry.path)}`);
                    const data = await res.json();
                    openFileInEditor(entry.path, data.content);
                } catch {
                    console.error('Failed to read file:', entry.path);
                }
            });
            container.appendChild(item);
        }
    });
}

// ═══════════════════════════════════════
// AI Panel — Model Download Buttons
// ═══════════════════════════════════════
function initAIPanel() {
    const panel = document.getElementById('ai-panel-content');
    panel.innerHTML = `
        <div class="ai-section-title">Models</div>

        <div class="model-card" id="model-foreman">
            <div class="model-card-header">
                <span class="model-card-name">🫀 The Pulse (Foreman)</span>
                <span class="model-card-status not-downloaded" id="status-foreman">Not Downloaded</span>
            </div>
            <div class="model-card-info">TinyLlama 1.1B — CPU • Always On</div>
            <div class="model-card-meta">~640 MB • Logic router, keystroke predictions, terminal supervision</div>
            <button class="model-download-btn" id="btn-download-foreman"
                onclick="downloadModel('foreman', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')">
                ⬇ Download TinyLlama 1.1B
            </button>
            <div class="download-progress" id="progress-foreman">
                <div class="download-progress-bar" id="progress-bar-foreman"></div>
            </div>
            <div class="download-progress-text" id="progress-text-foreman">0%</div>
        </div>

        <div class="model-card" id="model-logicgate">
            <div class="model-card-header">
                <span class="model-card-name">🧠 Logic-Gate</span>
                <span class="model-card-status not-downloaded" id="status-logicgate">Not Downloaded</span>
            </div>
            <div class="model-card-info">Qwen2.5-Coder 7B — GPU Static Buffer</div>
            <div class="model-card-meta">~4.5 GB • Ghost text, code completion, complexity routing</div>
            <button class="model-download-btn" id="btn-download-logicgate"
                onclick="downloadModel('logicgate', 'Qwen/Qwen2.5-Coder-7B')">
                ⬇ Download Qwen2.5-Coder 7B
            </button>
            <div class="download-progress" id="progress-logicgate">
                <div class="download-progress-bar" id="progress-bar-logicgate"></div>
            </div>
            <div class="download-progress-text" id="progress-text-logicgate">0%</div>
        </div>

        <div class="model-card" id="model-architect">
            <div class="model-card-header">
                <span class="model-card-name">🏗️ The Architect</span>
                <span class="model-card-status not-downloaded" id="status-architect">Not Downloaded</span>
            </div>
            <div class="model-card-info">DeepSeek-Coder 70B — Sharded via AirLLM</div>
            <div class="model-card-meta">~40 GB • Deep refactoring, architecture redesign</div>
            <button class="model-download-btn" id="btn-download-architect"
                onclick="downloadModel('architect', 'deepseek-ai/deepseek-coder-33b-instruct')">
                ⬇ Download DeepSeek-Coder 33B
            </button>
            <div class="download-progress" id="progress-architect">
                <div class="download-progress-bar" id="progress-bar-architect"></div>
            </div>
            <div class="download-progress-text" id="progress-text-architect">0%</div>
        </div>

        <div class="ai-section-title" style="margin-top:4px">Context Oracle</div>
        <div class="model-card">
            <div class="model-card-header">
                <span class="model-card-name">📚 Clara</span>
                <span class="model-card-status not-downloaded" id="status-clara">Not Indexed</span>
            </div>
            <div class="model-card-info">ChromaDB — Local RAG Context</div>
            <div class="model-card-meta">Indexes your repo for context-aware completions</div>
            <button class="model-download-btn" id="btn-index-clara"
                onclick="indexClara()">
                📂 Index Current Workspace
            </button>
        </div>

        <div class="ai-divider"></div>
        <div class="ai-section-title">Actions</div>
        <div class="ai-actions">
            <button class="ai-action-btn" disabled id="btn-ai-ghost">
                👻 Ghost Text <span style="color:var(--text-secondary);margin-left:auto;font-size:10px">Requires Logic-Gate</span>
            </button>
            <button class="ai-action-btn" disabled id="btn-ai-refactor">
                🏗️ Architect Refactor <span style="color:var(--text-secondary);margin-left:auto;font-size:10px">Requires Architect</span>
            </button>
            <button class="ai-action-btn" disabled id="btn-ai-explain">
                🧠 Explain Code <span style="color:var(--text-secondary);margin-left:auto;font-size:10px">Requires Logic-Gate</span>
            </button>
        </div>
    `;
}

async function downloadModel(tier, repoId) {
    const btn = document.getElementById(`btn-download-${tier}`);
    const status = document.getElementById(`status-${tier}`);
    const progress = document.getElementById(`progress-${tier}`);
    const progressBar = document.getElementById(`progress-bar-${tier}`);
    const progressText = document.getElementById(`progress-text-${tier}`);

    btn.disabled = true;
    btn.classList.add('downloading');
    btn.textContent = '⏳ Downloading...';
    status.className = 'model-card-status downloading';
    status.textContent = 'Downloading...';
    progress.classList.add('active');
    progressText.classList.add('active');

    try {
        const res = await fetch(`${API_BASE}/api/v1/models/download`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ tier, repo_id: repoId }),
        });

        if (!res.ok) {
            throw new Error('Download request failed');
        }

        // Poll progress
        const pollProgress = setInterval(async () => {
            try {
                const pRes = await fetch(`${API_BASE}/api/v1/models/status/${tier}`);
                const pData = await pRes.json();

                progressBar.style.width = pData.progress + '%';
                progressText.textContent = `${pData.progress}% — ${pData.message || ''}`;

                if (pData.status === 'ready') {
                    clearInterval(pollProgress);
                    btn.classList.remove('downloading');
                    btn.classList.add('downloaded');
                    btn.textContent = '✓ Downloaded';
                    status.className = 'model-card-status downloaded';
                    status.textContent = 'Ready';
                    progress.classList.remove('active');
                    progressText.classList.remove('active');
                    enableAIActions(tier);
                } else if (pData.status === 'error') {
                    clearInterval(pollProgress);
                    btn.disabled = false;
                    btn.classList.remove('downloading');
                    btn.textContent = '⬇ Retry Download';
                    status.className = 'model-card-status not-downloaded';
                    status.textContent = 'Error';
                    progressText.textContent = pData.message || 'Download failed';
                }
            } catch {
                // Keep polling
            }
        }, 2000);

    } catch (err) {
        btn.disabled = false;
        btn.classList.remove('downloading');
        btn.textContent = '⬇ Retry Download';
        status.className = 'model-card-status not-downloaded';
        status.textContent = 'Failed';
        progress.classList.remove('active');
        progressText.textContent = 'Backend not reachable. Start the server first.';
        progressText.classList.add('active');
    }
}

async function indexClara() {
    const btn = document.getElementById('btn-index-clara');
    const status = document.getElementById('status-clara');
    btn.disabled = true;
    btn.textContent = '⏳ Indexing...';
    status.textContent = 'Indexing...';
    status.className = 'model-card-status downloading';

    try {
        await fetch(`${API_BASE}/api/v1/clara/index`, { method: 'POST' });
        btn.textContent = '✓ Indexed';
        btn.classList.add('downloaded');
        status.textContent = 'Indexed';
        status.className = 'model-card-status downloaded';
    } catch {
        btn.disabled = false;
        btn.textContent = '📂 Retry Index';
        status.textContent = 'Error';
        status.className = 'model-card-status not-downloaded';
    }
}

function enableAIActions(tier) {
    if (tier === 'logicgate') {
        document.getElementById('btn-ai-ghost').disabled = false;
        document.getElementById('btn-ai-explain').disabled = false;
    }
    if (tier === 'architect') {
        document.getElementById('btn-ai-refactor').disabled = false;
    }
}

// ═══════════════════════════════════════
// Status Bar Polling
// ═══════════════════════════════════════
async function pollStatus() {
    try {
        const res = await fetch(`${API_BASE}/api/v1/status`);
        const data = await res.json();
        const statusEl = document.getElementById('status-ai-tier');
        const dot = statusEl.querySelector('.status-dot');

        if (data.connected !== false) {
            const tiers = [];
            if (data.foreman?.status === 'ready') tiers.push('F');
            if (data.logicGate?.status === 'ready') tiers.push('LG');
            if (data.architect?.status === 'ready') tiers.push('A');

            if (tiers.length > 0) {
                dot.className = 'status-dot online';
                statusEl.innerHTML = `<span class="status-dot online"></span> sealMega [${tiers.join('+')}]`;
            } else {
                dot.className = 'status-dot loading';
                statusEl.innerHTML = '<span class="status-dot loading"></span> sealMega: No Models';
            }

            // Update AI panel statuses
            updateModelCardStatus('foreman', data.foreman);
            updateModelCardStatus('logicgate', data.logicGate);
            updateModelCardStatus('architect', data.architect);
        }
    } catch {
        const statusEl = document.getElementById('status-ai-tier');
        statusEl.innerHTML = '<span class="status-dot offline"></span> sealMega: Offline';
    }

    setTimeout(pollStatus, 10000);
}

function updateModelCardStatus(tier, data) {
    if (!data) return;
    const status = document.getElementById(`status-${tier}`);
    const btn = document.getElementById(`btn-download-${tier}`);
    if (!status || !btn) return;

    if (data.status === 'ready') {
        status.className = 'model-card-status downloaded';
        status.textContent = 'Ready';
        btn.classList.add('downloaded');
        btn.textContent = '✓ Downloaded';
        btn.disabled = true;
        enableAIActions(tier);
    } else if (data.status === 'downloading') {
        status.className = 'model-card-status downloading';
        status.textContent = 'Downloading...';
    }
}

// ═══════════════════════════════════════
// Save File (Ctrl+S)
// ═══════════════════════════════════════
async function saveCurrentFile() {
    if (!activeFile || !monacoEditor) return;
    const content = monacoEditor.getValue();
    try {
        await fetch(`${API_BASE}/api/v1/files/write`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path: activeFile, content }),
        });
        // Remove modified indicator
        const tab = document.querySelector(`.tab[data-file="${CSS.escape(activeFile)}"]`);
        if (tab) {
            const nameEl = tab.querySelector('.tab-name');
            nameEl.textContent = nameEl.textContent.replace(/^● /, '');
        }
    } catch {
        console.error('Failed to save file');
    }
}

// ═══════════════════════════════════════
// Key Bindings
// ═══════════════════════════════════════
function initKeyBindings() {
    document.addEventListener('keydown', (e) => {
        // Ctrl+S — Save
        if (e.ctrlKey && e.key === 's') {
            e.preventDefault();
            saveCurrentFile();
        }
        // Ctrl+Shift+E — Explorer
        if (e.ctrlKey && e.shiftKey && e.key === 'E') {
            e.preventDefault();
            document.querySelector('[data-panel=explorer]').click();
        }
        // Ctrl+Shift+A — AI Panel
        if (e.ctrlKey && e.shiftKey && e.key === 'A') {
            e.preventDefault();
            document.querySelector('[data-panel=ai]').click();
        }
        // Ctrl+` — Toggle terminal
        if (e.ctrlKey && e.key === '`') {
            e.preventDefault();
            const panel = document.getElementById('bottom-panel');
            panel.style.display = panel.style.display === 'none' ? 'flex' : 'none';
        }
    });
}
