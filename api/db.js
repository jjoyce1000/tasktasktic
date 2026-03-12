const path = require('path');
const fs = require('fs');

const usePostgres = !!process.env.DATABASE_URL;
let db;

if (usePostgres) {
  const pg = require('./db-pg');
  db = {
    get: (sql, ...params) => pg.get(sql, ...params),
    all: (sql, ...params) => pg.all(sql, ...params),
    run: (sql, ...params) => pg.run(sql, ...params),
    transaction: (fn) => pg.transaction(fn)
  };
} else {
  const Database = require('better-sqlite3');
  const dataDir = path.join(__dirname, '..', 'data');
  if (!fs.existsSync(dataDir)) fs.mkdirSync(dataDir, { recursive: true });
  const sqlite = new Database(path.join(dataDir, 'todolist.db'));
  sqlite.exec(`
    CREATE TABLE IF NOT EXISTS users (id TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL, created_at TEXT DEFAULT (datetime('now')));
    CREATE TABLE IF NOT EXISTS courses (id TEXT PRIMARY KEY, user_id TEXT NOT NULL, name TEXT NOT NULL, description TEXT DEFAULT '', created_at TEXT DEFAULT (datetime('now')), FOREIGN KEY (user_id) REFERENCES users(id));
    CREATE TABLE IF NOT EXISTS tags (id TEXT PRIMARY KEY, user_id TEXT NOT NULL, name TEXT NOT NULL, color TEXT DEFAULT '#4a90d9', created_at TEXT DEFAULT (datetime('now')), FOREIGN KEY (user_id) REFERENCES users(id));
    CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, user_id TEXT NOT NULL, text TEXT NOT NULL, due_date TEXT, course_id TEXT DEFAULT '', category TEXT DEFAULT '', done INTEGER DEFAULT 0, parent_id TEXT, created_at TEXT DEFAULT (datetime('now')), FOREIGN KEY (user_id) REFERENCES users(id));
    CREATE INDEX IF NOT EXISTS idx_courses_user ON courses(user_id);
    CREATE INDEX IF NOT EXISTS idx_tags_user ON tags(user_id);
    CREATE INDEX IF NOT EXISTS idx_tasks_user ON tasks(user_id);
    CREATE INDEX IF NOT EXISTS idx_tasks_parent ON tasks(parent_id);
    CREATE TABLE IF NOT EXISTS user_preferences (user_id TEXT PRIMARY KEY, prefs TEXT DEFAULT '{}', FOREIGN KEY (user_id) REFERENCES users(id));
    CREATE TABLE IF NOT EXISTS usage_log (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, created_at TEXT DEFAULT (datetime('now')));
    CREATE INDEX IF NOT EXISTS idx_usage_log_user ON usage_log(user_id);
  `);
  try { sqlite.exec("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'user'"); } catch(e) { /* column may exist */ }
  try { sqlite.exec("ALTER TABLE courses ADD COLUMN color TEXT DEFAULT '#4a90d9'"); } catch(e) { /* column may exist */ }
  const txDb = {
    get: (s, ...p) => Promise.resolve(sqlite.prepare(s).get(...p)),
    all: (s, ...p) => Promise.resolve(sqlite.prepare(s).all(...p)),
    run: (s, ...p) => Promise.resolve(sqlite.prepare(s).run(...p))
  };
  db = {
    get: (sql, ...params) => Promise.resolve(sqlite.prepare(sql).get(...params)),
    all: (sql, ...params) => Promise.resolve(sqlite.prepare(sql).all(...params)),
    run: (sql, ...params) => Promise.resolve(sqlite.prepare(sql).run(...params)),
    transaction: async (fn) => {
      const txDb = {
        get: (s, ...p) => Promise.resolve(sqlite.prepare(s).get(...p)),
        all: (s, ...p) => Promise.resolve(sqlite.prepare(s).all(...p)),
        run: (s, ...p) => Promise.resolve(sqlite.prepare(s).run(...p))
      };
      return fn(txDb);
    }
  };
}

module.exports = db;
