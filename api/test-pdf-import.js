/**
 * Test script for PDF import API.
 * Run with: node api/test-pdf-import.js
 * Requires: API server running (npm run serve)
 */
const fs = require('fs');
const path = require('path');

const BASE = process.env.API_BASE || 'http://localhost:3001';

async function run() {
  console.log('Testing PDF import API at', BASE);
  let passed = 0;
  let failed = 0;

  // 1. Test 401 without auth
  try {
    const res = await fetch(BASE + '/api/import/pdf', { method: 'POST', body: new FormData() });
    if (res.status === 401) {
      console.log('  ✓ 401 without auth');
      passed++;
    } else {
      console.log('  ✗ Expected 401, got', res.status);
      failed++;
    }
  } catch (e) {
    console.log('  ✗ Request failed:', e.message);
    failed++;
  }

  // 2. Register test user
  let token;
  try {
    const reg = await fetch(BASE + '/api/auth/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email: 'test-pdf-' + Date.now() + '@test.com', password: 'test123' }),
    });
    const data = await reg.json().catch(() => ({}));
    if (reg.ok && data.token) {
      token = data.token;
      console.log('  ✓ Registered test user');
      passed++;
    } else {
      const login = await fetch(BASE + '/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: 'test-pdf@test.com', password: 'test123' }),
      });
      const loginData = await login.json().catch(() => ({}));
      if (login.ok && loginData.token) {
        token = loginData.token;
        console.log('  ✓ Logged in test user');
        passed++;
      } else {
        console.log('  ✗ Could not register or login');
        failed++;
      }
    }
  } catch (e) {
    console.log('  ✗ Auth failed:', e.message);
    failed++;
  }

  if (!token) {
    console.log('\nSkipping remaining tests (no token)');
    console.log('Result:', passed, 'passed,', failed, 'failed');
    process.exit(failed > 0 ? 1 : 0);
  }

  // 3. Test 400 with no file
  try {
    const form = new FormData();
    const res = await fetch(BASE + '/api/import/pdf', {
      method: 'POST',
      headers: { Authorization: 'Bearer ' + token },
      body: form,
    });
    const err = await res.json().catch(() => ({}));
    if (res.status === 400 && (err.error || '').includes('file')) {
      console.log('  ✓ 400 when no file uploaded');
      passed++;
    } else {
      console.log('  ✗ Expected 400 with file error, got', res.status, err);
      failed++;
    }
  } catch (e) {
    console.log('  ✗ No-file test failed:', e.message);
    failed++;
  }

  // 4. Test with minimal PDF (embedded or test.pdf file)
  let pdfBuffer;
  const pdfPath = path.join(__dirname, '..', 'test.pdf');
  if (fs.existsSync(pdfPath)) {
    pdfBuffer = fs.readFileSync(pdfPath);
  } else {
    // Minimal PDF with "Schedule Homework 1 Due 1/15/2026" text (base64)
    const minimalPdfB64 = 'JVBERi0xLjQKJeLjz9MKMSAwIG9iago8PAovVHlwZSAvQ2F0YWxvZwovUGFnZXMgMiAwIFIKPj4KZW5kb2JqCjIgMCBvYmoKPDwKL1R5cGUgL1BhZ2VzCi9LaWRzIFszIDAgUl0KL0NvdW50IDEKPj4KZW5kb2JqCjMgMCBvYmoKPDwKL1R5cGUgL1BhZ2UKL1BhcmVudCAyIDAgUgovTWVkaWFCb3ggWyAwIDAgNjEyIDc5MiBdCi9Db250ZW50cyA0IDAgUgovUmVzb3VyY2VzIDw8Ci9Gb250IDw8Ci9GMSA1IDAgUgo+Pgo+Pgo+PgplbmRvYmoKNCAwIG9iago8PAovTGVuZ3RoIDU0Cj4+CnN0cmVhbQpCVCAvRjEgMTIgVGYgMTAwIDcwMCBUZAooU2NoZWR1bGU6IEhvbWV3b3JrIDEgRHVlIDEvMTUvMjAyNikgVGoKRUQKZW5kc3RyZWFtCmVuZG9iago1IDAgb2JqCjw8Ci9UeXBlIC9Gb250Ci9TdWJ0eXBlIC9UeXBlMQovQmFzZUZvbnQgL0hlbHZldGljYQo+PgplbmRvYmoKeHJlZgowIDYKMDAwMDAwMDAwMCA2NTUzNSBmIAowMDAwMDAwMDA5IDAwMDAwIG4gCjAwMDAwMDAwNTggMDAwMDAgbiAKMDAwMDAwMDE0NyAwMDAwMCBuIAowMDAwMDAwMjU2IDAwMDAwIG4gCjAwMDAwMDAzMTAgMDAwMDAgbiAKdHJhaWxlcgo8PAovU2l6ZSA2Ci9Sb290IDEgMCBSCj4+CnN0YXJ0eHJlZgozOTQKJSVFT0YK';
    pdfBuffer = Buffer.from(minimalPdfB64, 'base64');
  }
  try {
    const form = new FormData();
    form.append('file', new Blob([pdfBuffer], { type: 'application/pdf' }), 'test.pdf');
    const res = await fetch(BASE + '/api/import/pdf', {
      method: 'POST',
      headers: { Authorization: 'Bearer ' + token },
      body: form,
    });
    const data = await res.json().catch(() => ({}));
    if (res.ok && Array.isArray(data.tasks)) {
      console.log('  ✓ PDF import returned', data.tasks.length, 'task(s)');
      passed++;
    } else {
      console.log('  ✗ PDF import failed:', res.status, data.error || data);
      failed++;
    }
  } catch (e) {
    console.log('  ✗ PDF import test failed:', e.message);
    failed++;
  }

  console.log('\nResult:', passed, 'passed,', failed, 'failed');
  process.exit(failed > 0 ? 1 : 0);
}

run().catch((e) => {
  console.error('Test error:', e);
  process.exit(1);
});
