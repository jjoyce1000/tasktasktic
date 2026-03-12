/**
 * Test PDF import with Anthropic API using LOCAL server.
 * Starts server with ANTHROPIC_API_KEY, runs import, verifies parser.
 * Usage: ANTHROPIC_API_KEY=sk-ant-... node api/test-pdf-anthropic-local.js [path-to-pdf]
 */
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

const PDF_PATH = process.argv[2] || path.join(__dirname, '..', 'test.pdf');
const PORT = 3099; // Use different port to avoid conflict with existing server
const BASE = `http://localhost:${PORT}`;

async function waitForServer(ms = 5000) {
  const start = Date.now();
  while (Date.now() - start < ms) {
    try {
      const r = await fetch(BASE + '/api/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: 'ping@test.com', password: 'x' }),
      });
      if (r.status === 200 || r.status === 400) return true;
    } catch (_) {}
    await new Promise((r) => setTimeout(r, 200));
  }
  return false;
}

async function run() {
  if (!process.env.ANTHROPIC_API_KEY) {
    console.error('ANTHROPIC_API_KEY is required. Set it before running:');
    console.error('  $env:ANTHROPIC_API_KEY="sk-ant-..."; node api/test-pdf-anthropic-local.js "path/to/file.pdf"');
    process.exit(1);
  }
  if (!fs.existsSync(PDF_PATH)) {
    console.error('File not found:', PDF_PATH);
    process.exit(1);
  }

  console.log('Starting local API server with ANTHROPIC_API_KEY...');
  const server = spawn('node', ['api/server.js'], {
    cwd: path.join(__dirname, '..'),
    env: { ...process.env, PORT: String(PORT) },
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  let serverOutput = '';
  server.stdout?.on('data', (d) => { serverOutput += d; });
  server.stderr?.on('data', (d) => { serverOutput += d; });

  const ready = await waitForServer();
  if (!ready) {
    server.kill('SIGTERM');
    console.error('Server failed to start');
    process.exit(1);
  }

  console.log('Server ready. Running PDF import test...');
  console.log('PDF:', PDF_PATH);
  console.log('');

  try {
    const pdfBuffer = fs.readFileSync(PDF_PATH);
    const filename = path.basename(PDF_PATH);

    const reg = await fetch(BASE + '/api/auth/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email: 'anthropic-test-' + Date.now() + '@test.com', password: 'test123' }),
    });
    const regData = await reg.json().catch(() => ({}));
    const token = regData.token;
    if (!token) {
      throw new Error('Could not register');
    }

    const form = new FormData();
    form.append('file', new Blob([pdfBuffer], { type: 'application/pdf' }), filename);
    const res = await fetch(BASE + '/api/import/pdf', {
      method: 'POST',
      headers: { Authorization: 'Bearer ' + token },
      body: form,
    });
    const data = await res.json().catch(() => ({}));

    if (!res.ok) {
      throw new Error('Import failed: ' + (data.error || res.status));
    }

    const parser = data.parser || 'unknown';
    const taskCount = Array.isArray(data.tasks) ? data.tasks.length : 0;

    console.log('--- Result ---');
    console.log('Parser:', parser);
    console.log('Task count:', taskCount);
    console.log('');
    if (parser === 'anthropic') {
      console.log('✓ Anthropic API is being used for PDF import');
      process.exitCode = 0;
    } else {
      console.log('✗ Anthropic API not used (parser:', parser, ')');
      console.log('  Check: ANTHROPIC_API_KEY valid? Credits available?');
      process.exitCode = 1;
    }
  } finally {
    server.kill('SIGTERM');
  }
}

run().catch((e) => {
  console.error(e);
  process.exit(1);
});
