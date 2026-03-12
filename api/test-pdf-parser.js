/**
 * Test which PDF parser the API uses (Anthropic vs regex).
 * Usage: node api/test-pdf-parser.js [path-to-pdf]
 * Uses API_BASE env (default: https://todo-list-app-rrjb.onrender.com)
 */
const fs = require('fs');
const path = require('path');

const BASE = process.env.API_BASE || 'https://todo-list-app-rrjb.onrender.com';
const PDF_PATH = process.argv[2] || path.join(__dirname, '..', 'test.pdf');

async function run() {
  if (!fs.existsSync(PDF_PATH)) {
    console.error('File not found:', PDF_PATH);
    process.exit(1);
  }
  const pdfBuffer = fs.readFileSync(PDF_PATH);
  const filename = path.basename(PDF_PATH);

  console.log('Testing PDF parser at', BASE);
  console.log('PDF:', PDF_PATH);
  console.log('');

  const reg = await fetch(BASE + '/api/auth/register', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email: 'parser-test-' + Date.now() + '@test.com', password: 'test123' }),
  });
  const regData = await reg.json().catch(() => ({}));
  let token = regData.token;
  if (!token) {
    const login = await fetch(BASE + '/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email: 'parser-test@test.com', password: 'test123' }),
    });
    const loginData = await login.json().catch(() => ({}));
    token = loginData.token;
  }
  if (!token) {
    console.error('Could not register or login');
    process.exit(1);
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
    console.error('Import failed:', res.status, data);
    process.exit(1);
  }

  const parser = data.parser || 'unknown';
  const taskCount = Array.isArray(data.tasks) ? data.tasks.length : 0;

  console.log('--- Result ---');
  console.log('Parser:', parser);
  console.log('Task count:', taskCount);
  console.log('');
  if (parser === 'anthropic') {
    console.log('✓ Using Anthropic API (Claude Sonnet 4.6)');
  } else if (parser === 'regex') {
    console.log('✗ Using regex fallback (ANTHROPIC_API_KEY not set or AI returned no tasks)');
  } else {
    console.log('? Parser field not in response (deploy latest code to see it)');
  }

  process.exit(parser === 'anthropic' ? 0 : 1);
}

run().catch((e) => {
  console.error(e);
  process.exit(1);
});
