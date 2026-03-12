const { Pool } = require('pg');

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl: process.env.DATABASE_URL?.includes('localhost') ? false : { rejectUnauthorized: false }
});

function toPgParams(sql) {
  let i = 0;
  return sql.replace(/\?/g, () => `$${++i}`);
}

async function get(sql, ...params) {
  const r = await pool.query(toPgParams(sql), params);
  return r.rows[0];
}

async function all(sql, ...params) {
  const r = await pool.query(toPgParams(sql), params);
  return r.rows;
}

async function run(sql, ...params) {
  const r = await pool.query(toPgParams(sql), params);
  return { changes: r.rowCount ?? 0 };
}

async function transaction(fn) {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');
    const db = {
      get: (sql, ...p) => client.query(toPgParams(sql, p), p).then(r => r.rows[0]),
      all: (sql, ...p) => client.query(toPgParams(sql, p), p).then(r => r.rows),
      run: (sql, ...p) => client.query(toPgParams(sql, p), p).then(r => ({ changes: r.rowCount ?? 0 }))
    };
    await fn(db);
    await client.query('COMMIT');
  } catch (e) {
    await client.query('ROLLBACK');
    throw e;
  } finally {
    client.release();
  }
}

async function init() {
  const client = await pool.connect();
  try {
    await client.query(`
      CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT DEFAULT 'user',
        created_at TIMESTAMPTZ DEFAULT NOW()
      );
      ALTER TABLE users ADD COLUMN IF NOT EXISTS role TEXT DEFAULT 'user';
      CREATE TABLE IF NOT EXISTS courses (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        name TEXT NOT NULL,
        description TEXT DEFAULT '',
        created_at TIMESTAMPTZ DEFAULT NOW()
      );
      ALTER TABLE courses ADD COLUMN IF NOT EXISTS color TEXT DEFAULT '#4a90d9';
      CREATE TABLE IF NOT EXISTS tags (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        name TEXT NOT NULL,
        color TEXT DEFAULT '#4a90d9',
        created_at TIMESTAMPTZ DEFAULT NOW()
      );
      CREATE TABLE IF NOT EXISTS tasks (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        text TEXT NOT NULL,
        due_date TEXT,
        course_id TEXT DEFAULT '',
        category TEXT DEFAULT '',
        done INTEGER DEFAULT 0,
        parent_id TEXT,
        created_at TIMESTAMPTZ DEFAULT NOW()
      );
      CREATE INDEX IF NOT EXISTS idx_courses_user ON courses(user_id);
      CREATE INDEX IF NOT EXISTS idx_tags_user ON tags(user_id);
      CREATE INDEX IF NOT EXISTS idx_tasks_user ON tasks(user_id);
      CREATE INDEX IF NOT EXISTS idx_tasks_parent ON tasks(parent_id);
      CREATE TABLE IF NOT EXISTS user_preferences (user_id TEXT PRIMARY KEY, prefs TEXT DEFAULT '{}');
      CREATE TABLE IF NOT EXISTS usage_log (id SERIAL PRIMARY KEY, user_id TEXT NOT NULL, created_at TIMESTAMPTZ DEFAULT NOW());
      CREATE INDEX IF NOT EXISTS idx_usage_log_user ON usage_log(user_id);
    `);
  } finally {
    client.release();
  }
}

module.exports = { get, all, run, transaction, init };
