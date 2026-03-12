/**
 * PDF to tasks parser for API.
 * Prefers Python pdf_to_csv.py (table extraction) when available for correct date alignment.
 * Falls back to calendar-grid, AI, then regex.
 */
const { PDFParse } = require('pdf-parse');
const { jsonrepair } = require('jsonrepair');
const auth = require('./auth');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');

let Anthropic;
try {
  ({ Anthropic } = require('@anthropic-ai/sdk'));
} catch {
  Anthropic = null;
}

// Anthropic API uses "claude-sonnet-4-6"; "anthropic.claude-sonnet-4-6" is AWS Bedrock format
function getPdfAiModel() {
  const raw = (process.env.PDF_AI_MODEL || 'claude-sonnet-4-6').trim();
  return raw.startsWith('anthropic.') ? raw.slice(10) : raw;
}

/**
 * Extract text from PDF buffer.
 */
async function extractPdfText(buffer) {
  const parser = new PDFParse({ data: buffer });
  try {
    const result = await parser.getText();
    await parser.destroy();
    return result?.text || '';
  } catch (e) {
    try { await parser.destroy(); } catch (_) {}
    throw e;
  }
}

function hasAnthropicKey() {
  const key = (process.env.ANTHROPIC_API_KEY || process.env.ANTHROPIC_AUTH_TOKEN || '').trim();
  return key.length > 0;
}

/**
 * Parse PDF content with Claude Sonnet 4.6 when ANTHROPIC_API_KEY is set.
 */
async function parseWithAi(text, filename) {
  if (!Anthropic || !hasAnthropicKey()) return null;

  const content = `Filename: ${filename || 'document.pdf'}\n\n--- Extracted Text ---\n${(text || '').slice(0, 90000)}`;

  const prompt = `Extract all tasks, assignments, due dates, exams, readings, and schedule entries from this syllabus/schedule PDF.
Return a JSON object with a "tasks" array. Each task must have:
- "task": short description (e.g. "Homework 1", "Exam 2", "Read Ch. 5")
- "date": YYYY-MM-DD format, or empty string if no date
- "course": course name/code (e.g. M156, Physics 111) or "General"

Rules:
- Infer year from document (Fall 2026, Spring 2026, etc.) when dates lack year
- One item per task; split "HW 1, 2, 3" into separate tasks if dates differ
- Skip headers, column labels, and meta text (e.g. "Week", "Monday", "Date")
- Use the course name from the document title, header, or filename (e.g. "M156-Syllabus.pdf" -> M156)
- Dates: prefer ISO YYYY-MM-DD; empty string if unknown
- Include ALL weeks from the start: calendar grids often have Mon–Fri columns; the first date in each row is Monday (e.g. 1/12). Do NOT skip Week 1 or the first week's content.
- CRITICAL – Date alignment in schedule grids: When the document has a grid with date columns (e.g. Mon 1/12 | Tue 1/13 | Wed 1/14 | Thu 1/15 | Fri 1/16), each task belongs to the date of its COLUMN. The first content cell maps to Monday's date, the second to Tuesday's, the third to Wednesday's, etc. "Section 5.5 - Substitution" in the Friday column must get Friday's date (e.g. 1/16), NOT Monday's (1/12). Match each task to the date of the column it appears under.

Example output: {"tasks": [{"task": "Homework 1 due", "date": "2026-01-15", "course": "M156"}, ...]}`;

  try {
    const client = new Anthropic();
    const resp = await client.messages.create({
      model: getPdfAiModel(),
      max_tokens: 8192,
      system: 'You extract structured task data from syllabi and schedules. Respond only with valid JSON.',
      messages: [{ role: 'user', content: `${prompt}\n\n${content}` }],
      temperature: 0.1,
    });
    const usage = resp.usage;
    if (usage) {
      const input = usage.input_tokens ?? 0;
      const output = usage.output_tokens ?? 0;
      const total = (usage.input_tokens ?? 0) + (usage.output_tokens ?? 0);
      console.log(`[PDF AI] tokens: input=${input} output=${output} total=${total}`);
    }
    const textBlock = resp.content?.find((b) => b.type === 'text');
    let raw = textBlock?.text;
    if (!raw) return null;
    raw = raw.trim();
    // Strip markdown code blocks (```json ... ``` or ``` ... ```)
    raw = raw.replace(/^```(?:json)?\s*\r?\n?/i, '').replace(/\r?\n?```\s*$/i, '').trim();
    // Fallback: if still has backticks or doesn't start with {, extract JSON object
    if (raw.includes('`') || !raw.startsWith('{')) {
      const start = raw.indexOf('{');
      const end = raw.lastIndexOf('}');
      if (start >= 0 && end > start) raw = raw.slice(start, end + 1);
    }
    let data;
    try {
      data = JSON.parse(raw);
    } catch (parseErr) {
      if (/unexpected end of json input|unexpected token|unexpected end/i.test(parseErr.message)) {
        try {
          const repaired = jsonrepair(raw);
          data = JSON.parse(repaired);
        } catch (repairErr) {
          const preview = raw.length > 400 ? raw.slice(0, 200) + '...[truncated]...' + raw.slice(-200) : raw;
          console.warn('[PDF AI] JSON parse failed, repair failed:', repairErr.message, '| raw preview:', preview);
          throw parseErr;
        }
      } else {
        throw parseErr;
      }
    }
    const tasks = data.tasks;
    if (!Array.isArray(tasks)) return null;
    return tasks.map((t) => ({
      id: auth.genId(),
      text: String(t.task || '').trim(),
      dueDate: String(t.date || '').trim(),
      courseId: '',
      category: 'School',
      tag: String(t.course || 'General').trim() || 'General',
      done: false,
      parentId: '',
    })).filter((t) => t.text.length > 0);
  } catch (err) {
    console.warn('PDF parseWithAi failed, falling back to regex:', err.message);
    return null;
  }
}

/**
 * Simple regex-based fallback when Claude is not available.
 */
function parseWithRegex(text, filename) {
  const items = [];
  const yearHint = extractYearHint(text, filename);
  const course = extractCourse(text, filename);

  const dateRe = /\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:,?\s+\d{4})?/gi;
  const dueRe = /(?:due|exam|deadline)\s*(?:by|date)?\s*[:\s]*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:,?\s+\d{4})?)/gi;

  const noise = new Set(['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday', 'week', 'topic', 'date', 'readings', 'assessment', 'homework due', 'lab']);
  const seen = new Set();

  function normDate(s) {
    if (!s || !s.trim()) return '';
    const m = s.match(/(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{2,4})/);
    if (m) {
      let [, mo, d, y] = m;
      y = parseInt(y, 10);
      if (y < 100) y = y < 50 ? 2000 + y : 1900 + y;
      if (y < 2024) y = yearHint;
      mo = parseInt(mo, 10);
      d = parseInt(d, 10);
      if (mo >= 1 && mo <= 12 && d >= 1 && d <= 31) return `${y}-${String(mo).padStart(2, '0')}-${String(d).padStart(2, '0')}`;
    }
    const monthRe = /(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(\d{1,2})(?:,?\s+(\d{4}))?/i;
    const mm = s.match(monthRe);
    if (mm) {
      const y = mm[3] ? parseInt(mm[3], 10) : yearHint;
      try {
        const dt = new Date(`${mm[1]} ${mm[2]} ${y}`);
        if (!isNaN(dt.getTime())) return dt.toISOString().slice(0, 10);
      } catch (_) {}
    }
    return '';
  }

  let currentDate = '';
  let dateBlockFirstDate = null; // First date in current date row (e.g. Mon–Fri) – use for content that follows
  const lines = (text || '').split(/\r?\n/).filter((l) => l.trim());

  for (const line of lines) {
    dueRe.lastIndex = 0;
    dateRe.lastIndex = 0;
    const dueMatch = dueRe.exec(line);
    if (dueMatch) {
      currentDate = normDate(dueMatch[1]) || currentDate;
      dateBlockFirstDate = null; // "due" breaks date block
    }

    const dateMatches = [];
    let m;
    while ((m = dateRe.exec(line)) !== null) {
      const d = normDate(m[0]);
      if (d) dateMatches.push(d);
    }

    let taskPart = line.replace(dateRe, '').replace(dueRe, '').replace(/\s+/g, ' ').trim();
    taskPart = taskPart.replace(/^(?:readings?|topic|assessment|homework|lab)\s*[:\-]\s*/i, '');
    const tl = (taskPart || '').toLowerCase();
    const isDateOnlyLine = dateMatches.length > 0 && (!taskPart || taskPart.length < 2 || noise.has(tl));

    if (isDateOnlyLine) {
      if (dateBlockFirstDate === null) dateBlockFirstDate = dateMatches[0];
      currentDate = dateBlockFirstDate;
      continue;
    }
    dateBlockFirstDate = null;
    if (dateMatches.length > 0) currentDate = dateMatches[0]; // Content line with inline date

    if (!taskPart || taskPart.length < 2) continue;
    if (noise.has(tl) || tl.startsWith('http') || /^page\s+\d+$/i.test(taskPart)) continue;
    const key = `${tl}|${currentDate}`;
    if (seen.has(key)) continue;
    seen.add(key);
    items.push({
      id: auth.genId(),
      text: taskPart,
      dueDate: currentDate,
      courseId: '',
      category: 'School',
      tag: course,
      done: false,
      parentId: '',
    });
  }
  return items;
}

function extractYearHint(text, filename) {
  const currentYear = new Date().getFullYear();
  const header = ((text || '') + ' ' + (filename || '')).slice(0, 1200);
  const m = header.match(/(?:Fall|Spring|Summer|Winter|Academic\s+Year)\s+(20\d{2})/i) ||
    header.match(/(20\d{2})[-–]\s*(20\d{2})/) ||
    header.match(/\b(20\d{2})\b/);
  if (m) {
    const y = parseInt(m[1] || m[2] || m[0], 10);
    if (y >= 1990 && y <= currentYear + 2) return y;
  }
  return currentYear;
}

function extractCourse(text, filename) {
  const first = ((text || '') + ' ' + (filename || '')).slice(0, 1200);
  const m = first.match(/(?:course|syllabus)\s*(?:title|name|number)?\s*[:\-]\s*([^\n]{3,80})/i) ||
    first.match(/(?:Physics|PHYS|Math|MATH|M\d{2,4}|CS|History|HIST|ENG|BIO)\s*\d{2,4}[A-Z]?/i) ||
    first.match(/[A-Z]{2,6}\s*\d{3}[A-Z]?/);
  if (m) return m[1]?.trim() || m[0]?.trim() || 'General';
  const fn = (filename || '').replace(/Copy of | - Sheet1/gi, '').trim();
  const stem = fn.replace(/\.(pdf|csv)$/i, '').trim();
  const fm = stem.match(/(Physics\s*111|M\d{2,4}|MATH\s*\d{2,4}|[A-Z]{2,6}\s*\d{3}[A-Z]?)/i);
  if (fm) return fm[1].trim();
  // Fallback: use filename stem as course/tag (e.g. "M156-Syllabus" -> "M156-Syllabus")
  const generic = ['', 'syllabus', 'schedule', 'calendar', 'document', 'import', 'pdf_import'];
  if (stem && !generic.includes(stem.toLowerCase()) && stem.length < 80) return stem;
  return 'General';
}

/**
 * Parse PDF using Python pdf_to_csv.py (table extraction for correct date alignment).
 * Returns tasks or null if Python unavailable or fails.
 */
async function parseWithPython(buffer, filename) {
  const scriptPath = path.join(__dirname, '..', 'pdf_to_csv.py');
  if (!fs.existsSync(scriptPath)) return null;

  const tmpDir = require('os').tmpdir();
  const tmpPdf = path.join(tmpDir, `pdf_import_${Date.now()}_${(Math.random() * 1e9) | 0}.pdf`);
  const tmpCsv = path.join(tmpDir, path.basename(tmpPdf, '.pdf') + '_schedule.csv');

  try {
    fs.writeFileSync(tmpPdf, buffer);
    await new Promise((resolve, reject) => {
      const proc = spawn('python', [scriptPath, tmpPdf, tmpCsv, filename || ''], {
        stdio: ['ignore', 'pipe', 'pipe'],
      });
      let stderr = '';
      proc.stderr?.on('data', (d) => { stderr += d.toString(); });
      proc.on('close', (code) => {
        if (code === 0) resolve();
        else reject(new Error(stderr || `python exited ${code}`));
      });
      proc.on('error', reject);
    });

    if (!fs.existsSync(tmpCsv)) return null;
    const csv = fs.readFileSync(tmpCsv, 'utf8');
    const lines = csv.split(/\r?\n/).filter((l) => l.trim());
    if (lines.length < 2) return null;

    const header = lines[0].toLowerCase();
    const taskIdx = header.includes('task') ? header.split(',').findIndex((h) => h.includes('task')) : 2;
    const dateIdx = header.includes('due') ? header.split(',').findIndex((h) => h.includes('due')) : 3;
    const tagIdx = header.includes('tag') ? header.split(',').findIndex((h) => h.includes('tag')) : 1;

    function parseCsvLine(line) {
      const out = [];
      let cur = '';
      let inQ = false;
      for (let i = 0; i < line.length; i++) {
        const c = line[i];
        if (c === '"') {
          inQ = !inQ;
        } else if (!inQ && c === ',') {
          out.push(cur);
          cur = '';
        } else {
          cur += c;
        }
      }
      out.push(cur);
      return out;
    }

    function mdyToIso(s) {
      if (!s || !s.trim()) return '';
      const m = s.trim().match(/^(\d{1,2})\/(\d{1,2})\/(\d{4})$/);
      if (m) {
        const [, mo, d, y] = m;
        return `${y}-${mo.padStart(2, '0')}-${d.padStart(2, '0')}`;
      }
      return '';
    }

    function splitConcatenatedTasks(text) {
      const parts = [];
      // Split on: Calc N Review - , Just in Time - , Section N, Quiz N, etc.
      // Do NOT split on Graphs & Derivatives (keeps "Quiz 1\nGraphs & Derivatives" as 1 task)
      const re = /\b(?:Calc \d+ Review - |Just in Time - |Section \d+\.?\d*(?:\s*\(cont'd\))?|Quiz \d+|WA due:|Last Day to|Calc \d|EXAM \d|Basic Integration|Area Between|Volumes?|Chapter \d|Syllabus\b|MLK Jr\. Day|\(No classes\))/gi;
      let lastEnd = 0;
      let m;
      while ((m = re.exec(text)) !== null) {
        if (m.index > lastEnd) {
          const before = text.slice(lastEnd, m.index).trim();
          if (before.length > 2) parts.push(before);
        }
        lastEnd = m.index;
      }
      const rest = text.slice(lastEnd).trim();
      if (rest.length > 2) parts.push(rest);
      if (parts.length === 0 && text.trim().length > 2) parts.push(text.trim());
      return parts;
    }

    const tasks = [];
    for (let i = 1; i < lines.length; i++) {
      const row = parseCsvLine(lines[i]);
      const taskText = (row[taskIdx] || '').replace(/^"|"$/g, '').replace(/""/g, '"').trim();
      const dueDate = mdyToIso((row[dateIdx] || '').replace(/^"|"$/g, ''));
      const tag = (row[tagIdx] || '').replace(/^"|"$/g, '').trim();
      if (!taskText || !dueDate) continue;
      const parts = splitConcatenatedTasks(taskText);
      for (const part of parts) {
        if (part.length > 2) {
          tasks.push({
            id: auth.genId(),
            text: part,
            dueDate,
            courseId: '',
            category: 'School',
            tag: tag || 'General',
            done: false,
            parentId: '',
          });
        }
      }
    }
    return tasks.length > 0 ? tasks : null;
  } catch (err) {
    return null;
  } finally {
    try { fs.unlinkSync(tmpPdf); } catch (_) {}
    try { fs.unlinkSync(tmpCsv); } catch (_) {}
  }
}

/**
 * Parse calendar grid from text: date row + content rows with column alignment.
 * Splits lines by 2+ spaces or tabs to detect columns. Maps each content cell to its column's date.
 */
function parseCalendarGridFromText(text, filename) {
  const yearHint = extractYearHint(text, filename);
  const course = extractCourse(text, filename);
  const items = [];
  const seen = new Set();
  const dateRe = /\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:,?\s+\d{4})?/gi;
  const noise = new Set(['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'week', 'keew', 'mon', 'tue', 'wed', 'thu', 'fri']);

  function normDate(s) {
    if (!s || !s.trim()) return '';
    const m = String(s).match(/(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{2,4})/);
    if (m) {
      let [, mo, d, y] = m;
      y = parseInt(y, 10);
      if (y < 100) y = y < 50 ? 2000 + y : 1900 + y;
      if (y < 2024) y = yearHint;
      mo = parseInt(mo, 10);
      d = parseInt(d, 10);
      if (mo >= 1 && mo <= 12 && d >= 1 && d <= 31) return `${y}-${String(mo).padStart(2, '0')}-${String(d).padStart(2, '0')}`;
    }
    const monthRe = /(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(\d{1,2})(?:,?\s+(\d{4}))?/i;
    const mm = String(s).match(monthRe);
    if (mm) {
      const y = mm[3] ? parseInt(mm[3], 10) : yearHint;
      try {
        const dt = new Date(`${mm[1]} ${mm[2]} ${y}`);
        if (!isNaN(dt.getTime())) return dt.toISOString().slice(0, 10);
      } catch (_) {}
    }
    return '';
  }

  function looksLikeDate(cell) {
    if (!cell || !cell.trim()) return false;
    return /^\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}$/.test(cell.trim()) ||
      /^\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2}$/.test(cell.trim()) ||
      /^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:,?\s+\d{4})?$/i.test(cell.trim());
  }

  function addTask(task, dateIso) {
    task = (task || '').replace(dateRe, '').replace(/\s+/g, ' ').trim();
    task = task.replace(/^(?:readings?|topic|assessment|homework|lab)\s*[:\-]\s*/i, '');
    if (!task || task.length < 2) return;
    const tl = task.toLowerCase();
    if (noise.has(tl) || tl.startsWith('http') || /^page\s+\d+$/i.test(task)) return;
    if (/\d+\s*keew/i.test(task)) return;
    const key = `${tl}|${dateIso}`;
    if (seen.has(key)) return;
    seen.add(key);
    items.push({
      id: auth.genId(),
      text: task,
      dueDate: dateIso || '',
      courseId: '',
      category: 'School',
      tag: course,
      done: false,
      parentId: '',
    });
  }

  const lines = (text || '').split(/\r?\n/).filter((l) => l.trim());
  let dateCols = null;
  let verticalColIndex = 0;
  let lastTask = null;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    let cells = line.split(/\t+|\s{2,}/).map((c) => c.trim()).filter(Boolean);

    const dateCount = cells.filter((c) => looksLikeDate(c)).length;
    const isDateRow = dateCount >= 3 && cells.some((c) => normDate(c));

    if (isDateRow) {
      dateCols = [];
      for (let j = 0; j < cells.length; j++) {
        const iso = normDate(cells[j]);
        if (iso) dateCols.push({ index: j, date: iso });
      }
      verticalColIndex = 0;
      lastTask = null;
      continue;
    }

    if (dateCols && dateCols.length >= 3) {
      const isWeekLabel = /^WEEK\s+\d+$/i.test(line.trim()) || /^\d+\s*keew$/i.test(line.trim());
      if (isWeekLabel) continue;

      if (cells.length >= dateCols.length) {
        for (const { index, date } of dateCols) {
          const cell = cells[index] || '';
          if (cell && !looksLikeDate(cell)) addTask(cell, date);
        }
      } else if (cells.length === 1 && dateCols[verticalColIndex % dateCols.length]) {
        const { date } = dateCols[verticalColIndex % dateCols.length];
        const cell = cells[0] || line.trim();
        if (cell && !looksLikeDate(cell)) {
          const isContinuation = lastTask &&
            /section\s*\d|^\d+\.\d+/i.test(lastTask.text) &&
            /^[a-z\-]+$/i.test(cell) && cell.length < 25;
          if (isContinuation) {
            lastTask.text = lastTask.text + ' - ' + cell;
            verticalColIndex++;
          } else {
            addTask(cell, date);
            lastTask = items[items.length - 1];
            verticalColIndex++;
          }
        }
      }
    }
  }

  return items.length > 0 ? items : null;
}

/**
 * Parse PDF buffer into tasks.
 * @param {Buffer} buffer - PDF file buffer
 * @param {string} filename - Original filename for hints
 * @returns {Promise<{tasks: Array, parser: string}>} tasks and parser used ('anthropic' or 'regex')
 */
async function parsePdfToTasks(buffer, filename = 'document.pdf') {
  let tasks = await parseWithPython(buffer, filename);
  if (tasks && tasks.length > 0) {
    return { tasks, parser: 'python' };
  }

  const text = await extractPdfText(buffer);
  if (!text || text.trim().length < 10) {
    throw new Error('Could not extract text from PDF or PDF appears empty.');
  }

  tasks = parseCalendarGridFromText(text, filename);
  if (tasks && tasks.length > 0) {
    return { tasks, parser: 'calendar-grid' };
  }

  tasks = await parseWithAi(text, filename);
  if (tasks && tasks.length > 0) {
    return { tasks, parser: 'anthropic' };
  }
  tasks = parseWithRegex(text, filename);
  if (!tasks || tasks.length === 0) {
    throw new Error('No tasks found in PDF.');
  }
  return { tasks, parser: 'regex' };
}

module.exports = { parsePdfToTasks, extractPdfText };
