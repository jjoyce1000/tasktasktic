/**
 * Upload a PDF to the Render API and output the parsed tasks.
 * Usage: node api/test-pdf-render.js <path-to-pdf>
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

  console.log('API:', BASE);
  console.log('PDF:', PDF_PATH);
  console.log('');

  // Register test user
  const reg = await fetch(BASE + '/api/auth/register', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email: 'compare-' + Date.now() + '@test.com', password: 'test123' }),
  });
  const regData = await reg.json().catch(() => ({}));
  let token = regData.token;
  if (!token) {
    const login = await fetch(BASE + '/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email: 'compare@test.com', password: 'test123' }),
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

  console.log(JSON.stringify(data, null, 2));
}

run().catch((e) => {
  console.error(e);
  process.exit(1);
});
