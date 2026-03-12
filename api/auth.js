const jwt = require('jsonwebtoken');
const bcrypt = require('bcryptjs');
const db = require('./db');

const JWT_SECRET = process.env.JWT_SECRET || 'todolist-secret-change-in-production';
const SALT_ROUNDS = 10;

function genId() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2);
}

function hashPassword(password) {
  return bcrypt.hashSync(password, SALT_ROUNDS);
}

function verifyPassword(password, hash) {
  return bcrypt.compareSync(password, hash);
}

function createToken(userId) {
  return jwt.sign({ userId }, JWT_SECRET, { expiresIn: '7d' });
}

function verifyToken(token) {
  try {
    const decoded = jwt.verify(token, JWT_SECRET);
    return decoded.userId;
  } catch {
    return null;
  }
}

async function register(email, password) {
  const existing = await db.get('SELECT id FROM users WHERE email = ?', email.toLowerCase());
  if (existing) throw new Error('Email already registered');
  const id = genId();
  const hash = hashPassword(password);
  await db.run('INSERT INTO users (id, email, password_hash) VALUES (?, ?, ?)', id, email.toLowerCase(), hash);
  return { id, email: email.toLowerCase(), token: createToken(id) };
}

async function login(email, password) {
  const user = await db.get('SELECT id, email, password_hash FROM users WHERE email = ?', email.toLowerCase());
  if (!user || !verifyPassword(password, user.password_hash)) throw new Error('Invalid email or password');
  return { id: user.id, email: user.email, token: createToken(user.id) };
}

module.exports = { genId, hashPassword, verifyPassword, createToken, verifyToken, register, login };
