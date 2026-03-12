/**
 * Test that Anthropic API is working.
 * Usage: node api/test-anthropic.js
 * Requires: ANTHROPIC_API_KEY in environment
 */
const { Anthropic } = require('@anthropic-ai/sdk');

async function run() {
  if (!process.env.ANTHROPIC_API_KEY) {
    console.error('ANTHROPIC_API_KEY is not set');
    console.error('Set it before running: $env:ANTHROPIC_API_KEY="sk-ant-..."; node api/test-anthropic.js');
    process.exit(1);
  }

  console.log('Testing Anthropic API...');
  try {
    const client = new Anthropic();
    const resp = await client.messages.create({
      model: 'claude-sonnet-4-6',
      max_tokens: 64,
      messages: [{ role: 'user', content: 'Reply with exactly: OK' }],
    });
    const text = resp.content?.find((b) => b.type === 'text')?.text;
    if (text && text.trim().includes('OK')) {
      console.log('✓ Anthropic API is working');
      console.log('  Model:', resp.model);
      console.log('  Response:', text.trim().slice(0, 80));
      process.exit(0);
    } else {
      console.error('Unexpected response:', text);
      process.exit(1);
    }
  } catch (err) {
    console.error('✗ Anthropic API failed:', err.message);
    if (err.status) console.error('  Status:', err.status);
    process.exit(1);
  }
}

run();
