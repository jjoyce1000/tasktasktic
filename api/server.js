const express = require('express');
const cors = require('cors');
const path = require('path');
const multer = require('multer');
const db = require('./db');
const auth = require('./auth');
const { parsePdfToTasks } = require('./pdf-parser');

const usePostgres = !!process.env.DATABASE_URL;
const app = express();
const PORT = process.env.PORT || 3001;

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 10 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    if (file.mimetype === 'application/pdf') cb(null, true);
    else cb(new Error('Only PDF files are allowed'));
  },
});

app.use(cors());
app.use(express.json({ limit: '10mb' }));

function authMiddleware(req, res, next) {
  const header = req.headers.authorization;
  const token = header && header.startsWith('Bearer ') ? header.slice(7) : null;
  const userId = token ? auth.verifyToken(token) : null;
  if (!userId) return res.status(401).json({ error: 'Unauthorized' });
  req.userId = userId;
  next();
}

async function adminMiddleware(req, res, next) {
  const user = await db.get('SELECT id, role FROM users WHERE id = ?', req.userId);
  if (!user || user.role !== 'admin') return res.status(403).json({ error: 'Admin access required' });
  next();
}

async function ensureAdminByEmail(email) {
  const adminEmail = process.env.ADMIN_EMAIL;
  if (!adminEmail || !email) return;
  await db.run('UPDATE users SET role = ? WHERE LOWER(email) = LOWER(?)', 'admin', adminEmail.trim());
}

function taskRowToObj(row) {
  return {
    id: row.id,
    text: row.text,
    dueDate: row.due_date || '',
    courseId: row.course_id || '',
    category: row.category || '',
    done: !!row.done,
    parentId: row.parent_id || '',
  };
}

app.post('/api/auth/register', async (req, res) => {
  try {
    const { email, password } = req.body;
    if (!email || !password) return res.status(400).json({ error: 'Email and password required' });
    const result = await auth.register(email, password);
    await ensureAdminByEmail(email);
    if (usePostgres) db.run('INSERT INTO usage_log (user_id) VALUES (?)', result.id).catch(() => {});
    const user = await db.get('SELECT id, email, role FROM users WHERE id = ?', result.id);
    res.json({ user: { id: user.id, email: user.email, role: user.role || 'user' }, token: result.token });
  } catch (e) {
    res.status(400).json({ error: e.message });
  }
});

app.post('/api/auth/login', async (req, res) => {
  try {
    const { email, password } = req.body;
    if (!email || !password) return res.status(400).json({ error: 'Email and password required' });
    const result = await auth.login(email, password);
    await ensureAdminByEmail(email);
    if (usePostgres) db.run('INSERT INTO usage_log (user_id) VALUES (?)', result.id).catch(() => {});
    const user = await db.get('SELECT id, email, role FROM users WHERE id = ?', result.id);
    res.json({ user: { id: user.id, email: user.email, role: user.role || 'user' }, token: result.token });
  } catch (e) {
    res.status(401).json({ error: e.message });
  }
});

app.get('/api/me', authMiddleware, async (req, res) => {
  const user = await db.get('SELECT id, email, role FROM users WHERE id = ?', req.userId);
  if (!user) return res.status(404).json({ error: 'User not found' });
  res.json({ id: user.id, email: user.email, role: user.role || 'user' });
});

app.get('/api/courses', authMiddleware, async (req, res) => {
  const rows = await db.all('SELECT id, name, description, color FROM courses WHERE user_id = ?', req.userId);
  res.json(rows.map(r => ({ id: r.id, name: r.name, description: r.description || '', color: r.color || null })));
});

app.post('/api/courses', authMiddleware, async (req, res) => {
  const { name, description } = req.body;
  if (!name) return res.status(400).json({ error: 'Name required' });
  const id = auth.genId();
  await db.run('INSERT INTO courses (id, user_id, name, description) VALUES (?, ?, ?, ?)', id, req.userId, name, description || '');
  res.json({ id, name, description: description || '' });
});

app.put('/api/courses/:id', authMiddleware, async (req, res) => {
  const { name, description } = req.body;
  const r = await db.run('UPDATE courses SET name = ?, description = ? WHERE id = ? AND user_id = ?', name || '', description || '', req.params.id, req.userId);
  if (r.changes === 0) return res.status(404).json({ error: 'Not found' });
  res.json({ ok: true });
});

app.delete('/api/courses/:id', authMiddleware, async (req, res) => {
  await db.run('DELETE FROM courses WHERE id = ? AND user_id = ?', req.params.id, req.userId);
  await db.run('UPDATE tasks SET course_id = ? WHERE course_id = ? AND user_id = ?', '', req.params.id, req.userId);
  res.json({ ok: true });
});

app.get('/api/tags', authMiddleware, async (req, res) => {
  const rows = await db.all('SELECT id, name, color FROM tags WHERE user_id = ?', req.userId);
  res.json(rows);
});

app.post('/api/tags', authMiddleware, async (req, res) => {
  const { name, color } = req.body;
  if (!name) return res.status(400).json({ error: 'Name required' });
  const existing = await db.get('SELECT id FROM tags WHERE user_id = ? AND LOWER(name) = LOWER(?)', req.userId, name);
  if (existing) return res.status(400).json({ error: 'Tag already exists' });
  const id = auth.genId();
  await db.run('INSERT INTO tags (id, user_id, name, color) VALUES (?, ?, ?, ?)', id, req.userId, name, color || '#4a90d9');
  res.json({ id, name, color: color || '#4a90d9' });
});

app.put('/api/tags/:id', authMiddleware, async (req, res) => {
  const { name, color } = req.body;
  const old = await db.get('SELECT name FROM tags WHERE id = ? AND user_id = ?', req.params.id, req.userId);
  if (!old) return res.status(404).json({ error: 'Not found' });
  if (name !== undefined) await db.run('UPDATE tags SET name = ? WHERE id = ? AND user_id = ?', name, req.params.id, req.userId);
  if (color !== undefined) await db.run('UPDATE tags SET color = ? WHERE id = ? AND user_id = ?', color, req.params.id, req.userId);
  if (name !== undefined && name !== old.name) await db.run('UPDATE tasks SET category = ? WHERE category = ? AND user_id = ?', name, old.name, req.userId);
  res.json({ ok: true });
});

app.delete('/api/tags/:id', authMiddleware, async (req, res) => {
  const tag = await db.get('SELECT name FROM tags WHERE id = ? AND user_id = ?', req.params.id, req.userId);
  if (tag) {
    await db.run('DELETE FROM tags WHERE id = ? AND user_id = ?', req.params.id, req.userId);
    await db.run('UPDATE tasks SET category = ? WHERE category = ? AND user_id = ?', '', tag.name, req.userId);
  }
  res.json({ ok: true });
});

app.get('/api/tasks', authMiddleware, async (req, res) => {
  const rows = await db.all('SELECT id, text, due_date, course_id, category, done, parent_id FROM tasks WHERE user_id = ?', req.userId);
  res.json(rows.map(taskRowToObj));
});

app.get('/api/data', authMiddleware, async (req, res) => {
  if (usePostgres) {
    db.run('INSERT INTO usage_log (user_id) VALUES (?)', req.userId).catch(() => {});
  }
  const courseRows = await db.all('SELECT id, name, description, color FROM courses WHERE user_id = ?', req.userId);
  const courses = courseRows.map(r => ({ id: r.id, name: r.name, description: r.description || '', color: r.color || undefined }));
  const tags = await db.all('SELECT id, name, color FROM tags WHERE user_id = ?', req.userId);
  const taskRows = await db.all('SELECT id, text, due_date, course_id, category, done, parent_id FROM tasks WHERE user_id = ?', req.userId);
  const tasks = taskRows.map(taskRowToObj);
  const prefsRow = await db.get('SELECT prefs FROM user_preferences WHERE user_id = ?', req.userId);
  const preferences = prefsRow?.prefs ? JSON.parse(prefsRow.prefs) : {};
  res.json({ courses, tags, tasks, preferences });
});

app.get('/api/preferences', authMiddleware, async (req, res) => {
  const row = await db.get('SELECT prefs FROM user_preferences WHERE user_id = ?', req.userId);
  const preferences = row?.prefs ? JSON.parse(row.prefs) : {};
  res.json(preferences);
});

app.put('/api/preferences', authMiddleware, async (req, res) => {
  const { preferences } = req.body;
  if (!preferences || typeof preferences !== 'object') return res.status(400).json({ error: 'preferences object required' });
  const prefsJson = JSON.stringify(preferences);
  await db.run('INSERT INTO user_preferences (user_id, prefs) VALUES (?, ?) ON CONFLICT(user_id) DO UPDATE SET prefs = excluded.prefs', req.userId, prefsJson);
  res.json({ ok: true });
});

app.post('/api/tasks', authMiddleware, async (req, res) => {
  const { text, dueDate, courseId, category, done, parentId } = req.body;
  if (!text) return res.status(400).json({ error: 'Text required' });
  const id = auth.genId();
  await db.run('INSERT INTO tasks (id, user_id, text, due_date, course_id, category, done, parent_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
    id, req.userId, text, dueDate || null, courseId || '', category || '', done ? 1 : 0, parentId || null);
  const row = await db.get('SELECT * FROM tasks WHERE id = ?', id);
  res.json(taskRowToObj(row));
});

app.put('/api/tasks/:id', authMiddleware, async (req, res) => {
  const { text, dueDate, courseId, category, done, parentId } = req.body;
  const r = await db.run('UPDATE tasks SET text = COALESCE(?, text), due_date = ?, course_id = COALESCE(?, course_id), category = COALESCE(?, category), done = COALESCE(?, done), parent_id = ? WHERE id = ? AND user_id = ?',
    text || undefined, dueDate || null, courseId ?? undefined, category ?? undefined, done !== undefined ? (done ? 1 : 0) : undefined, parentId || null, req.params.id, req.userId);
  if (r.changes === 0) return res.status(404).json({ error: 'Not found' });
  const row = await db.get('SELECT * FROM tasks WHERE id = ?', req.params.id);
  res.json(taskRowToObj(row));
});

app.delete('/api/tasks/:id', authMiddleware, async (req, res) => {
  await db.run('DELETE FROM tasks WHERE id = ? AND user_id = ?', req.params.id, req.userId);
  await db.run('DELETE FROM tasks WHERE parent_id = ? AND user_id = ?', req.params.id, req.userId);
  res.json({ ok: true });
});

app.post('/api/sync', authMiddleware, async (req, res) => {
  try {
    const { courses: cs, tags: ts, tasks: tks, merge } = req.body;
    if (!Array.isArray(cs) || !Array.isArray(ts) || !Array.isArray(tks)) {
      return res.status(400).json({ error: 'courses, tags, tasks arrays required' });
    }
    const uid = req.userId;
    const useMerge = !!merge;
    await db.transaction(async (tx) => {
      if (!useMerge) {
        await tx.run('DELETE FROM courses WHERE user_id = ?', uid);
        await tx.run('DELETE FROM tags WHERE user_id = ?', uid);
        await tx.run('DELETE FROM tasks WHERE user_id = ?', uid);
      }
      const courseUpsert = useMerge && usePostgres
        ? 'INSERT INTO courses (id, user_id, name, description, color) VALUES (?, ?, ?, ?, ?) ON CONFLICT (id) DO UPDATE SET user_id=EXCLUDED.user_id, name=EXCLUDED.name, description=EXCLUDED.description, color=EXCLUDED.color'
        : useMerge
          ? 'INSERT OR REPLACE INTO courses (id, user_id, name, description, color) VALUES (?, ?, ?, ?, ?)'
          : null;
      const tagUpsert = useMerge && usePostgres
        ? 'INSERT INTO tags (id, user_id, name, color) VALUES (?, ?, ?, ?) ON CONFLICT (id) DO UPDATE SET user_id=EXCLUDED.user_id, name=EXCLUDED.name, color=EXCLUDED.color'
        : useMerge
          ? 'INSERT OR REPLACE INTO tags (id, user_id, name, color) VALUES (?, ?, ?, ?)'
          : null;
      for (const c of cs) {
        await tx.run(courseUpsert || 'INSERT INTO courses (id, user_id, name, description, color) VALUES (?, ?, ?, ?, ?)', c.id, uid, c.name, c.description || '', c.color || '#4a90d9');
      }
      for (const t of ts) {
        await tx.run(tagUpsert || 'INSERT INTO tags (id, user_id, name, color) VALUES (?, ?, ?, ?)', t.id, uid, t.name, t.color || '#4a90d9');
      }
      const taskUpsert = useMerge && usePostgres
        ? 'INSERT INTO tasks (id, user_id, text, due_date, course_id, category, done, parent_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT (id) DO UPDATE SET user_id=EXCLUDED.user_id, text=EXCLUDED.text, due_date=EXCLUDED.due_date, course_id=EXCLUDED.course_id, category=EXCLUDED.category, done=EXCLUDED.done, parent_id=EXCLUDED.parent_id'
        : useMerge
          ? 'INSERT OR REPLACE INTO tasks (id, user_id, text, due_date, course_id, category, done, parent_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)'
          : null;
      for (const t of tks) {
        await tx.run(taskUpsert || 'INSERT INTO tasks (id, user_id, text, due_date, course_id, category, done, parent_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
          t.id, uid, t.text, t.dueDate || null, t.courseId || '', t.category || '', t.done ? 1 : 0, t.parentId || null);
      }
    });
    res.json({ ok: true });
  } catch (err) {
    console.error('Sync error:', err);
    res.status(500).json({ error: err.message || 'Sync failed' });
  }
});

app.post('/api/import/pdf', authMiddleware, (req, res, next) => {
  upload.single('file')(req, res, (err) => {
    if (err) {
      if (err.code === 'LIMIT_FILE_SIZE') return res.status(400).json({ error: 'PDF file too large (max 10MB)' });
      return res.status(400).json({ error: err.message || 'Invalid file upload' });
    }
    next();
  });
}, async (req, res) => {
  try {
    if (!req.file || !req.file.buffer) {
      return res.status(400).json({ error: 'No PDF file uploaded. Use form field "file".' });
    }
    const filename = req.file.originalname || 'document.pdf';
    const { tasks, parser } = await parsePdfToTasks(req.file.buffer, filename);
    console.log(`[PDF import] filename=${filename} parser=${parser} tasks=${tasks.length}`);
    res.json({ tasks, parser });
  } catch (err) {
    console.error('PDF import error:', err);
    res.status(400).json({ error: err.message || 'PDF parsing failed' });
  }
});

app.get('/api/admin/stats', authMiddleware, adminMiddleware, async (req, res) => {
  if (!usePostgres) return res.status(404).json({ error: 'Not found' });
  try {
    const totalRow = await db.get('SELECT COUNT(DISTINCT user_id) as n FROM usage_log');
    const rows = await db.all(
      'SELECT u.email, COUNT(l.id) as usage_count FROM usage_log l JOIN users u ON l.user_id = u.id GROUP BY l.user_id, u.email ORDER BY usage_count DESC'
    );
    res.json({
      totalAccounts: totalRow?.n ?? 0,
      accounts: rows.map(r => ({ email: r.email, usageCount: r.usage_count }))
    });
  } catch (err) {
    console.error('Stats error:', err);
    res.status(500).json({ error: 'Stats failed' });
  }
});

app.use(express.static(path.join(__dirname, '..')));

app.get('/admin', (req, res) => {
  res.sendFile(path.join(__dirname, '..', 'admin.html'));
});

app.get('*', (req, res) => {
  if (!req.path.startsWith('/api')) {
    res.sendFile(path.join(__dirname, '..', 'index.html'));
  } else {
    res.status(404).json({ error: 'Not found' });
  }
});

async function start() {
  if (process.env.DATABASE_URL) {
    const pg = require('./db-pg');
    await pg.init();
    console.log('Using PostgreSQL database');
  }
  app.listen(PORT, () => {
    console.log(`To-Do List API running at http://localhost:${PORT}`);
    console.log('Press Ctrl+C to stop');
  });
}

start().catch(err => {
  console.error('Failed to start:', err);
  process.exit(1);
});
